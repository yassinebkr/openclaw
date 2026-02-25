/**
 * Local patch for streaming compaction progress.
 *
 * This duplicates the SDK's compact() and generateSummary() logic but uses
 * streamSimple instead of completeSimple to enable progress streaming.
 *
 * Falls back to regular SDK compact() if streaming fails.
 */

import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { Context, Model, Api, SimpleStreamOptions } from "@mariozechner/pi-ai";
import { streamSimple, completeSimple } from "@mariozechner/pi-ai";
import {
  type CompactionResult,
  type CompactionSettings,
  compact as sdkCompact,
  generateSummary as sdkGenerateSummary,
  convertToLlm,
} from "@mariozechner/pi-coding-agent";

/**
 * Progress event emitted during streaming compaction
 */
export interface CompactionProgress {
  phase: "summarizing" | "prefix_summary" | "finalizing";
  tokensGenerated: number;
  estimatedTotal: number;
}

/**
 * Progress callback type
 */
export type ProgressCallback = (progress: CompactionProgress) => void;

/**
 * Input parameters for streaming compaction
 */
export interface CompactionPreparation {
  firstKeptEntryId: string;
  messagesToSummarize: AgentMessage[];
  turnPrefixMessages: AgentMessage[];
  isSplitTurn: boolean;
  tokensBefore: number;
  previousSummary?: string;
  fileOps: FileOperations;
  settings: CompactionSettings;
}

/**
 * File operations tracking
 */
export interface FileOperations {
  read: Set<string>;
  written: Set<string>;
  edited: Set<string>;
}

// ============================================================================
// Constants and utilities extracted from SDK (not exported)
// ============================================================================

const SUMMARIZATION_SYSTEM_PROMPT = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI coding assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.`;

const SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

const UPDATE_SUMMARIZATION_PROMPT = `The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

const TURN_PREFIX_SUMMARIZATION_PROMPT = `This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix.`;

/**
 * Serialize LLM messages to text for summarization.
 * Duplicated from SDK utils (not exported).
 */
function serializeConversation(messages: AgentMessage[]): string {
  const parts: string[] = [];

  for (const msg of messages) {
    if (msg.role === "user") {
      const content =
        typeof msg.content === "string"
          ? msg.content
          : msg.content
              .filter((c) => c.type === "text")
              .map((c) => (c as any).text)
              .join("");
      if (content) parts.push(`[User]: ${content}`);
    } else if (msg.role === "assistant") {
      const assistant = msg as any;
      const textParts: string[] = [];
      const thinkingParts: string[] = [];
      const toolCalls: string[] = [];

      for (const block of assistant.content) {
        if (block.type === "text") {
          textParts.push(block.text);
        } else if (block.type === "thinking") {
          thinkingParts.push(block.thinking);
        } else if (block.type === "toolCall") {
          const args = block.arguments;
          const argsStr = Object.entries(args)
            .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
            .join(", ");
          toolCalls.push(`${block.name}(${argsStr})`);
        }
      }

      if (thinkingParts.length > 0) {
        parts.push(`[Assistant thinking]: ${thinkingParts.join("\n")}`);
      }
      if (textParts.length > 0) {
        parts.push(`[Assistant]: ${textParts.join("\n")}`);
      }
      if (toolCalls.length > 0) {
        parts.push(`[Assistant tool calls]: ${toolCalls.join("; ")}`);
      }
    } else if (msg.role === "toolResult") {
      const content = (msg as any).content
        .filter((c: any) => c.type === "text")
        .map((c: any) => c.text)
        .join("");
      if (content) {
        parts.push(`[Tool result]: ${content}`);
      }
    }
  }

  return parts.join("\n\n");
}

/**
 * Create file operations tracking
 */
function createFileOps(): FileOperations {
  return {
    read: new Set(),
    written: new Set(),
    edited: new Set(),
  };
}

/**
 * Compute final file lists from file operations.
 */
function computeFileLists(fileOps: FileOperations): {
  readFiles: string[];
  modifiedFiles: string[];
} {
  const modified = new Set([...fileOps.edited, ...fileOps.written]);
  const readOnly = [...fileOps.read].filter((f) => !modified.has(f)).sort();
  const modifiedFiles = [...modified].sort();
  return { readFiles: readOnly, modifiedFiles };
}

/**
 * Format file operations as XML tags for summary.
 */
function formatFileOperations(readFiles: string[], modifiedFiles: string[]): string {
  const sections: string[] = [];

  if (readFiles.length > 0) {
    sections.push(`<read-files>\n${readFiles.join("\n")}\n</read-files>`);
  }

  if (modifiedFiles.length > 0) {
    sections.push(`<modified-files>\n${modifiedFiles.join("\n")}\n</modified-files>`);
  }

  if (sections.length === 0) return "";
  return `\n\n${sections.join("\n\n")}`;
}

// ============================================================================
// Streaming implementation
// ============================================================================

/**
 * Generate a summary with streaming progress events.
 * Mirrors the SDK's generateSummary() but uses streamSimple.
 */
export async function generateSummaryWithProgress<TApi extends Api>(
  currentMessages: AgentMessage[],
  model: Model<TApi>,
  reserveTokens: number,
  apiKey: string | undefined,
  signal: AbortSignal | undefined,
  customInstructions: string | undefined,
  previousSummary: string | undefined,
  onProgress: ProgressCallback,
): Promise<string> {
  const maxTokens = Math.floor(0.8 * reserveTokens);

  // Use update prompt if we have a previous summary, otherwise initial prompt
  let basePrompt = previousSummary ? UPDATE_SUMMARIZATION_PROMPT : SUMMARIZATION_PROMPT;
  if (customInstructions) {
    basePrompt = `${basePrompt}\n\nAdditional focus: ${customInstructions}`;
  }

  // Convert to LLM messages first (handles custom types)
  const llmMessages = convertToLlm(currentMessages);
  const conversationText = serializeConversation(llmMessages);

  // Build the prompt with conversation wrapped in tags
  let promptText = `<conversation>\n${conversationText}\n</conversation>\n\n`;
  if (previousSummary) {
    promptText += `<previous-summary>\n${previousSummary}\n</previous-summary>\n\n`;
  }
  promptText += basePrompt;

  const summarizationMessages = [
    {
      role: "user",
      content: [{ type: "text", text: promptText }],
      timestamp: Date.now(),
    },
  ] as AgentMessage[];

  const context: Context = {
    systemPrompt: SUMMARIZATION_SYSTEM_PROMPT,
    messages: summarizationMessages,
  };

  const options: SimpleStreamOptions = {
    maxTokens,
    signal,
    apiKey,
    reasoning: "high",
  };

  // Use streaming to get progress events
  const eventStream = streamSimple(model, context, options);

  let tokensGenerated = 0;
  const textParts: string[] = [];

  try {
    for await (const event of eventStream) {
      if (signal?.aborted) {
        throw new Error("Summarization aborted");
      }

      if (event.type === "text_delta") {
        tokensGenerated += Math.ceil(event.delta.length / 4); // Rough token estimate
        textParts.push(event.delta);

        onProgress({
          phase: "summarizing",
          tokensGenerated,
          estimatedTotal: maxTokens,
        });
      } else if (event.type === "error") {
        throw new Error(`Summarization failed: ${event.error.errorMessage || "Unknown error"}`);
      } else if (event.type === "done") {
        // Collect final text content
        const finalTextContent = event.message.content
          .filter((c) => c.type === "text")
          .map((c) => (c as any).text)
          .join("\n");
        return finalTextContent;
      }
    }

    // If we reach here, stream ended without 'done' event
    return textParts.join("");
  } catch (error) {
    // If streaming fails, fall back to non-streaming
    console.warn("Streaming summarization failed, falling back to non-streaming:", error);
    return await sdkGenerateSummary(
      currentMessages,
      model,
      reserveTokens,
      apiKey,
      signal,
      customInstructions,
      previousSummary,
    );
  }
}

/**
 * Generate a summary for a turn prefix with streaming progress.
 */
async function generateTurnPrefixSummaryWithProgress<TApi extends Api>(
  messages: AgentMessage[],
  model: Model<TApi>,
  reserveTokens: number,
  apiKey: string | undefined,
  signal: AbortSignal | undefined,
  onProgress: ProgressCallback,
): Promise<string> {
  const maxTokens = Math.floor(0.5 * reserveTokens); // Smaller budget for turn prefix

  const llmMessages = convertToLlm(messages);
  const conversationText = serializeConversation(llmMessages);
  const promptText = `<conversation>\n${conversationText}\n</conversation>\n\n${TURN_PREFIX_SUMMARIZATION_PROMPT}`;

  const summarizationMessages = [
    {
      role: "user",
      content: [{ type: "text", text: promptText }],
      timestamp: Date.now(),
    },
  ] as AgentMessage[];

  const context: Context = {
    systemPrompt: SUMMARIZATION_SYSTEM_PROMPT,
    messages: summarizationMessages,
  };

  const options: SimpleStreamOptions = {
    maxTokens,
    signal,
    apiKey,
  };

  const eventStream = streamSimple(model, context, options);

  let tokensGenerated = 0;
  const textParts: string[] = [];

  try {
    for await (const event of eventStream) {
      if (signal?.aborted) {
        throw new Error("Turn prefix summarization aborted");
      }

      if (event.type === "text_delta") {
        tokensGenerated += Math.ceil(event.delta.length / 4); // Rough token estimate
        textParts.push(event.delta);

        onProgress({
          phase: "prefix_summary",
          tokensGenerated,
          estimatedTotal: maxTokens,
        });
      } else if (event.type === "error") {
        throw new Error(
          `Turn prefix summarization failed: ${event.error.errorMessage || "Unknown error"}`,
        );
      } else if (event.type === "done") {
        const finalTextContent = event.message.content
          .filter((c) => c.type === "text")
          .map((c) => (c as any).text)
          .join("\n");
        return finalTextContent;
      }
    }

    return textParts.join("");
  } catch (error) {
    // If streaming fails, fall back to non-streaming SDK call
    console.warn(
      "Streaming turn prefix summarization failed, falling back to non-streaming:",
      error,
    );

    // Fall back to completeSimple (similar to SDK's generateTurnPrefixSummary)
    const response = await completeSimple(model, context, options);
    if (response.stopReason === "error") {
      throw new Error(
        `Turn prefix summarization failed: ${response.errorMessage || "Unknown error"}`,
      );
    }
    return response.content
      .filter((c) => c.type === "text")
      .map((c) => (c as any).text)
      .join("\n");
  }
}

/**
 * Main streaming compaction function.
 * Mirrors the SDK's compact() but with progress streaming.
 */
export async function compactWithProgress<TApi extends Api>(
  preparation: CompactionPreparation,
  model: Model<TApi>,
  apiKey: string | undefined,
  customInstructions: string | undefined,
  signal: AbortSignal | undefined,
  onProgress: ProgressCallback,
): Promise<CompactionResult> {
  const {
    firstKeptEntryId,
    messagesToSummarize,
    turnPrefixMessages,
    isSplitTurn,
    tokensBefore,
    previousSummary,
    fileOps,
    settings,
  } = preparation;

  try {
    // Generate summaries (can be parallel if both needed) and merge into one
    let summary: string;

    if (isSplitTurn && turnPrefixMessages.length > 0) {
      // Generate both summaries in parallel
      const [historyResult, turnPrefixResult] = await Promise.all([
        messagesToSummarize.length > 0
          ? generateSummaryWithProgress(
              messagesToSummarize,
              model,
              settings.reserveTokens,
              apiKey,
              signal,
              customInstructions,
              previousSummary,
              onProgress,
            )
          : Promise.resolve("No prior history."),
        generateTurnPrefixSummaryWithProgress(
          turnPrefixMessages,
          model,
          settings.reserveTokens,
          apiKey,
          signal,
          onProgress,
        ),
      ]);

      // Merge into single summary
      summary = `${historyResult}\n\n---\n\n**Turn Context (split turn):**\n\n${turnPrefixResult}`;
    } else {
      // Just generate history summary
      summary = await generateSummaryWithProgress(
        messagesToSummarize,
        model,
        settings.reserveTokens,
        apiKey,
        signal,
        customInstructions,
        previousSummary,
        onProgress,
      );
    }

    // Finalization phase
    onProgress({
      phase: "finalizing",
      tokensGenerated: 0,
      estimatedTotal: 100, // Arbitrary small number for finalization
    });

    // Compute file lists and append to summary
    const { readFiles, modifiedFiles } = computeFileLists(fileOps);
    summary += formatFileOperations(readFiles, modifiedFiles);

    if (!firstKeptEntryId) {
      throw new Error("First kept entry has no UUID - session may need migration");
    }

    return {
      summary,
      firstKeptEntryId,
      tokensBefore,
      details: { readFiles, modifiedFiles },
    };
  } catch (error) {
    // If anything fails, fall back to the regular SDK compact
    console.warn("Streaming compaction failed, falling back to SDK compact():", error);
    return await sdkCompact(preparation, model, apiKey, customInstructions, signal);
  }
}
