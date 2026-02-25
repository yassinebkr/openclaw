/**
 * Tests for streaming compaction functionality.
 */

import type {
  AgentMessage,
  AssistantMessageEventStream,
  Model,
  Api,
  AssistantMessageEvent,
} from "@mariozechner/pi-ai";
import { streamSimple, completeSimple } from "@mariozechner/pi-ai";
import {
  compact as sdkCompact,
  generateSummary as sdkGenerateSummary,
} from "@mariozechner/pi-coding-agent";
import { describe, it, expect, vi, beforeEach, type MockedFunction } from "vitest";
import {
  compactWithProgress,
  generateSummaryWithProgress,
  type CompactionProgress,
  type CompactionPreparation,
} from "./compact-streaming.js";

// Mock the external modules
vi.mock("@mariozechner/pi-ai", () => ({
  streamSimple: vi.fn(),
  completeSimple: vi.fn(),
}));

vi.mock("@mariozechner/pi-coding-agent", () => ({
  compact: vi.fn(),
  generateSummary: vi.fn(),
  convertToLlm: vi.fn((messages) => messages), // Simple passthrough for tests
}));

const mockStreamSimple = streamSimple as MockedFunction<typeof streamSimple>;
const mockCompleteSimple = completeSimple as MockedFunction<typeof completeSimple>;
const mockSdkCompact = sdkCompact as MockedFunction<typeof sdkCompact>;
const mockSdkGenerateSummary = sdkGenerateSummary as MockedFunction<typeof sdkGenerateSummary>;

describe("compact-streaming", () => {
  const mockModel = "anthropic/claude-3-5-sonnet" as Model<Api>;
  const mockApiKey = "test-api-key";

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("generateSummaryWithProgress", () => {
    const mockMessages: AgentMessage[] = [
      {
        role: "user",
        content: [{ type: "text", text: "Hello, can you help me with a project?" }],
        timestamp: Date.now(),
      },
      {
        role: "assistant",
        content: [{ type: "text", text: "Of course! What kind of project are you working on?" }],
        api: "anthropic-messages",
        provider: "anthropic",
        model: "claude-3-5-sonnet",
        usage: {
          input: 10,
          output: 15,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 25,
          cost: { input: 0.001, output: 0.002, cacheRead: 0, cacheWrite: 0, total: 0.003 },
        },
        stopReason: "stop",
        timestamp: Date.now(),
      } as AgentMessage,
    ];

    it("should call onProgress with text_delta events", async () => {
      const mockEvents: AssistantMessageEvent[] = [
        {
          type: "start",
          partial: {
            role: "assistant",
            content: [],
            api: "anthropic-messages",
            provider: "anthropic",
            model: "claude-3-5-sonnet",
            usage: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 0,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
            stopReason: "stop",
            timestamp: Date.now(),
          },
        },
        {
          type: "text_delta",
          contentIndex: 0,
          delta: "This is a summary",
          partial: {
            role: "assistant",
            content: [{ type: "text", text: "This is a summary" }],
            api: "anthropic-messages",
            provider: "anthropic",
            model: "claude-3-5-sonnet",
            usage: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 0,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
            stopReason: "stop",
            timestamp: Date.now(),
          },
        },
        {
          type: "text_delta",
          contentIndex: 0,
          delta: " of the conversation.",
          partial: {
            role: "assistant",
            content: [{ type: "text", text: "This is a summary of the conversation." }],
            api: "anthropic-messages",
            provider: "anthropic",
            model: "claude-3-5-sonnet",
            usage: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 0,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
            stopReason: "stop",
            timestamp: Date.now(),
          },
        },
        {
          type: "done",
          reason: "stop",
          message: {
            role: "assistant",
            content: [{ type: "text", text: "This is a summary of the conversation." }],
            api: "anthropic-messages",
            provider: "anthropic",
            model: "claude-3-5-sonnet",
            usage: {
              input: 20,
              output: 10,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 30,
              cost: { input: 0.002, output: 0.001, cacheRead: 0, cacheWrite: 0, total: 0.003 },
            },
            stopReason: "stop",
            timestamp: Date.now(),
          },
        },
      ];

      // Create a mock async iterator
      const mockIterator = {
        async *[Symbol.asyncIterator]() {
          for (const event of mockEvents) {
            yield event;
          }
        },
      };

      mockStreamSimple.mockReturnValue(mockIterator as AssistantMessageEventStream);

      const progressEvents: CompactionProgress[] = [];
      const onProgress = vi.fn((progress: CompactionProgress) => {
        progressEvents.push(progress);
      });

      const result = await generateSummaryWithProgress(
        mockMessages,
        mockModel,
        1000, // reserveTokens
        mockApiKey,
        undefined, // signal
        undefined, // customInstructions
        undefined, // previousSummary
        onProgress,
      );

      expect(result).toBe("This is a summary of the conversation.");
      expect(onProgress).toHaveBeenCalledTimes(2);
      expect(progressEvents).toHaveLength(2);

      // Check that progress events have the right structure and phases
      expect(progressEvents[0].phase).toBe("summarizing");
      expect(progressEvents[0].tokensGenerated).toBeGreaterThan(0);
      expect(progressEvents[0].estimatedTotal).toBe(Math.floor(0.8 * 1000));

      expect(progressEvents[1].phase).toBe("summarizing");
      expect(progressEvents[1].tokensGenerated).toBeGreaterThan(progressEvents[0].tokensGenerated);
      expect(progressEvents[1].estimatedTotal).toBe(Math.floor(0.8 * 1000));
    });

    it("should fallback to SDK generateSummary on streaming error", async () => {
      // Mock streamSimple to return an iterator that throws during iteration
      const mockIterator = {
        async *[Symbol.asyncIterator]() {
          throw new Error("Streaming failed");
        },
      };

      mockStreamSimple.mockReturnValue(mockIterator as AssistantMessageEventStream);

      // Mock SDK fallback
      mockSdkGenerateSummary.mockResolvedValue("Fallback summary");

      const onProgress = vi.fn();

      const result = await generateSummaryWithProgress(
        mockMessages,
        mockModel,
        1000,
        mockApiKey,
        undefined,
        undefined,
        undefined,
        onProgress,
      );

      expect(result).toBe("Fallback summary");
      expect(mockSdkGenerateSummary).toHaveBeenCalledWith(
        mockMessages,
        mockModel,
        1000,
        mockApiKey,
        undefined,
        undefined,
        undefined,
      );
      expect(onProgress).not.toHaveBeenCalled();
    });

    it("should handle abort signal", async () => {
      const abortController = new AbortController();
      const signal = abortController.signal;

      const mockEvents: AssistantMessageEvent[] = [
        {
          type: "text_delta",
          contentIndex: 0,
          delta: "Starting summary",
          partial: {
            role: "assistant",
            content: [{ type: "text", text: "Starting summary" }],
            api: "anthropic-messages",
            provider: "anthropic",
            model: "claude-3-5-sonnet",
            usage: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 0,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
            stopReason: "stop",
            timestamp: Date.now(),
          },
        },
      ];

      const mockIterator = {
        async *[Symbol.asyncIterator]() {
          for (const event of mockEvents) {
            // Abort during processing
            abortController.abort();
            yield event;
          }
        },
      };

      mockStreamSimple.mockReturnValue(mockIterator as AssistantMessageEventStream);
      mockSdkGenerateSummary.mockResolvedValue("Fallback after abort");

      const onProgress = vi.fn();

      const result = await generateSummaryWithProgress(
        mockMessages,
        mockModel,
        1000,
        mockApiKey,
        signal,
        undefined,
        undefined,
        onProgress,
      );

      // Should fallback due to abort
      expect(result).toBe("Fallback after abort");
      expect(mockSdkGenerateSummary).toHaveBeenCalled();
    });

    it("should handle custom instructions", async () => {
      const mockEvents: AssistantMessageEvent[] = [
        {
          type: "done",
          reason: "stop",
          message: {
            role: "assistant",
            content: [{ type: "text", text: "Custom summary with focus on testing" }],
            api: "anthropic-messages",
            provider: "anthropic",
            model: "claude-3-5-sonnet",
            usage: {
              input: 20,
              output: 10,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 30,
              cost: { input: 0.002, output: 0.001, cacheRead: 0, cacheWrite: 0, total: 0.003 },
            },
            stopReason: "stop",
            timestamp: Date.now(),
          },
        },
      ];

      const mockIterator = {
        async *[Symbol.asyncIterator]() {
          for (const event of mockEvents) {
            yield event;
          }
        },
      };

      mockStreamSimple.mockReturnValue(mockIterator as AssistantMessageEventStream);

      const onProgress = vi.fn();
      const customInstructions = "Focus on testing aspects";

      const result = await generateSummaryWithProgress(
        mockMessages,
        mockModel,
        1000,
        mockApiKey,
        undefined,
        customInstructions,
        undefined,
        onProgress,
      );

      expect(result).toBe("Custom summary with focus on testing");
      expect(mockStreamSimple).toHaveBeenCalledWith(
        mockModel,
        expect.objectContaining({
          systemPrompt: expect.any(String),
          messages: expect.arrayContaining([
            expect.objectContaining({
              content: expect.arrayContaining([
                expect.objectContaining({
                  text: expect.stringContaining(customInstructions),
                }),
              ]),
            }),
          ]),
        }),
        expect.objectContaining({
          maxTokens: Math.floor(0.8 * 1000),
          apiKey: mockApiKey,
          reasoning: "high",
        }),
      );
    });
  });

  describe("compactWithProgress", () => {
    const mockPreparation: CompactionPreparation = {
      firstKeptEntryId: "test-entry-id",
      messagesToSummarize: [
        {
          role: "user",
          content: [{ type: "text", text: "Test message" }],
          timestamp: Date.now(),
        },
      ],
      turnPrefixMessages: [],
      isSplitTurn: false,
      tokensBefore: 1000,
      previousSummary: undefined,
      fileOps: {
        read: new Set(["file1.ts"]),
        written: new Set(["file2.ts"]),
        edited: new Set(["file3.ts"]),
      },
      settings: {
        enabled: true,
        reserveTokens: 2000,
        keepRecentTokens: 5000,
      },
    };

    it("should complete normal compaction with progress events", async () => {
      const mockEvents: AssistantMessageEvent[] = [
        {
          type: "text_delta",
          contentIndex: 0,
          delta: "## Goal\nTest the application",
          partial: {
            role: "assistant",
            content: [{ type: "text", text: "## Goal\nTest the application" }],
            api: "anthropic-messages",
            provider: "anthropic",
            model: "claude-3-5-sonnet",
            usage: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 0,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
            stopReason: "stop",
            timestamp: Date.now(),
          },
        },
        {
          type: "done",
          reason: "stop",
          message: {
            role: "assistant",
            content: [
              {
                type: "text",
                text: "## Goal\nTest the application\n\n## Progress\n### Done\n- [x] Created test file",
              },
            ],
            api: "anthropic-messages",
            provider: "anthropic",
            model: "claude-3-5-sonnet",
            usage: {
              input: 50,
              output: 30,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 80,
              cost: { input: 0.005, output: 0.003, cacheRead: 0, cacheWrite: 0, total: 0.008 },
            },
            stopReason: "stop",
            timestamp: Date.now(),
          },
        },
      ];

      const mockIterator = {
        async *[Symbol.asyncIterator]() {
          for (const event of mockEvents) {
            yield event;
          }
        },
      };

      mockStreamSimple.mockReturnValue(mockIterator as AssistantMessageEventStream);

      const progressEvents: CompactionProgress[] = [];
      const onProgress = vi.fn((progress: CompactionProgress) => {
        progressEvents.push(progress);
      });

      const result = await compactWithProgress(
        mockPreparation,
        mockModel,
        mockApiKey,
        undefined,
        undefined,
        onProgress,
      );

      expect(result).toEqual({
        summary: expect.stringContaining("## Goal\nTest the application"),
        firstKeptEntryId: "test-entry-id",
        tokensBefore: 1000,
        details: {
          readFiles: ["file1.ts"],
          modifiedFiles: ["file2.ts", "file3.ts"],
        },
      });

      // Should have at least summarizing and finalizing phases
      expect(progressEvents.length).toBeGreaterThan(0);
      expect(progressEvents.some((p) => p.phase === "summarizing")).toBe(true);
      expect(progressEvents.some((p) => p.phase === "finalizing")).toBe(true);
    });

    it("should handle split turn scenario with parallel summaries", async () => {
      const splitTurnPreparation: CompactionPreparation = {
        ...mockPreparation,
        isSplitTurn: true,
        turnPrefixMessages: [
          {
            role: "user",
            content: [{ type: "text", text: "Start of long turn" }],
            timestamp: Date.now(),
          },
        ],
      };

      // Mock events for main summary
      const mainSummaryEvents: AssistantMessageEvent[] = [
        {
          type: "text_delta",
          contentIndex: 0,
          delta: "Main summary",
          partial: {
            role: "assistant",
            content: [{ type: "text", text: "Main summary" }],
            api: "anthropic-messages",
            provider: "anthropic",
            model: "claude-3-5-sonnet",
            usage: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 0,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
            stopReason: "stop",
            timestamp: Date.now(),
          },
        },
        {
          type: "done",
          reason: "stop",
          message: {
            role: "assistant",
            content: [{ type: "text", text: "Main summary content" }],
            api: "anthropic-messages",
            provider: "anthropic",
            model: "claude-3-5-sonnet",
            usage: {
              input: 50,
              output: 30,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 80,
              cost: { input: 0.005, output: 0.003, cacheRead: 0, cacheWrite: 0, total: 0.008 },
            },
            stopReason: "stop",
            timestamp: Date.now(),
          },
        },
      ];

      // Mock events for prefix summary
      const prefixSummaryEvents: AssistantMessageEvent[] = [
        {
          type: "text_delta",
          contentIndex: 0,
          delta: "Prefix context",
          partial: {
            role: "assistant",
            content: [{ type: "text", text: "Prefix context" }],
            api: "anthropic-messages",
            provider: "anthropic",
            model: "claude-3-5-sonnet",
            usage: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 0,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
            stopReason: "stop",
            timestamp: Date.now(),
          },
        },
        {
          type: "done",
          reason: "stop",
          message: {
            role: "assistant",
            content: [{ type: "text", text: "Prefix summary content" }],
            api: "anthropic-messages",
            provider: "anthropic",
            model: "claude-3-5-sonnet",
            usage: {
              input: 30,
              output: 20,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 50,
              cost: { input: 0.003, output: 0.002, cacheRead: 0, cacheWrite: 0, total: 0.005 },
            },
            stopReason: "stop",
            timestamp: Date.now(),
          },
        },
      ];

      // Mock streamSimple to return different iterators for different calls
      let callCount = 0;
      mockStreamSimple.mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          // First call - main summary
          return {
            async *[Symbol.asyncIterator]() {
              for (const event of mainSummaryEvents) {
                yield event;
              }
            },
          } as AssistantMessageEventStream;
        } else {
          // Second call - prefix summary
          return {
            async *[Symbol.asyncIterator]() {
              for (const event of prefixSummaryEvents) {
                yield event;
              }
            },
          } as AssistantMessageEventStream;
        }
      });

      const progressEvents: CompactionProgress[] = [];
      const onProgress = vi.fn((progress: CompactionProgress) => {
        progressEvents.push(progress);
      });

      const result = await compactWithProgress(
        splitTurnPreparation,
        mockModel,
        mockApiKey,
        undefined,
        undefined,
        onProgress,
      );

      // Should contain merged summary with turn context
      expect(result.summary).toContain("Main summary content");
      expect(result.summary).toContain("**Turn Context (split turn):**");
      expect(result.summary).toContain("Prefix summary content");

      // Should have both summarizing and prefix_summary phases
      expect(progressEvents.some((p) => p.phase === "summarizing")).toBe(true);
      expect(progressEvents.some((p) => p.phase === "prefix_summary")).toBe(true);
      expect(progressEvents.some((p) => p.phase === "finalizing")).toBe(true);
    });

    it("should fallback to SDK compact on error", async () => {
      // Mock streamSimple to throw an error
      mockStreamSimple.mockImplementation(() => {
        throw new Error("Streaming failed");
      });

      const mockSdkResult = {
        summary: "SDK fallback summary",
        firstKeptEntryId: "test-entry-id",
        tokensBefore: 1000,
        details: { readFiles: [], modifiedFiles: [] },
      };

      mockSdkCompact.mockResolvedValue(mockSdkResult);

      const onProgress = vi.fn();

      const result = await compactWithProgress(
        mockPreparation,
        mockModel,
        mockApiKey,
        undefined,
        undefined,
        onProgress,
      );

      expect(result).toEqual(mockSdkResult);
      expect(mockSdkCompact).toHaveBeenCalledWith(
        mockPreparation,
        mockModel,
        mockApiKey,
        undefined,
        undefined,
      );
    });

    it("should include file operations in summary", async () => {
      const mockEvents: AssistantMessageEvent[] = [
        {
          type: "done",
          reason: "stop",
          message: {
            role: "assistant",
            content: [{ type: "text", text: "Base summary content" }],
            api: "anthropic-messages",
            provider: "anthropic",
            model: "claude-3-5-sonnet",
            usage: {
              input: 50,
              output: 30,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 80,
              cost: { input: 0.005, output: 0.003, cacheRead: 0, cacheWrite: 0, total: 0.008 },
            },
            stopReason: "stop",
            timestamp: Date.now(),
          },
        },
      ];

      const mockIterator = {
        async *[Symbol.asyncIterator]() {
          for (const event of mockEvents) {
            yield event;
          }
        },
      };

      mockStreamSimple.mockReturnValue(mockIterator as AssistantMessageEventStream);

      const onProgress = vi.fn();

      const result = await compactWithProgress(
        mockPreparation,
        mockModel,
        mockApiKey,
        undefined,
        undefined,
        onProgress,
      );

      expect(result.summary).toContain("Base summary content");
      expect(result.summary).toContain("<read-files>");
      expect(result.summary).toContain("file1.ts");
      expect(result.summary).toContain("<modified-files>");
      expect(result.summary).toContain("file2.ts");
      expect(result.summary).toContain("file3.ts");

      expect(result.details).toEqual({
        readFiles: ["file1.ts"],
        modifiedFiles: ["file2.ts", "file3.ts"],
      });
    });

    it("should handle custom instructions", async () => {
      const customInstructions = "Focus on performance optimizations";

      const mockEvents: AssistantMessageEvent[] = [
        {
          type: "done",
          reason: "stop",
          message: {
            role: "assistant",
            content: [{ type: "text", text: "Performance-focused summary" }],
            api: "anthropic-messages",
            provider: "anthropic",
            model: "claude-3-5-sonnet",
            usage: {
              input: 50,
              output: 30,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 80,
              cost: { input: 0.005, output: 0.003, cacheRead: 0, cacheWrite: 0, total: 0.008 },
            },
            stopReason: "stop",
            timestamp: Date.now(),
          },
        },
      ];

      const mockIterator = {
        async *[Symbol.asyncIterator]() {
          for (const event of mockEvents) {
            yield event;
          }
        },
      };

      mockStreamSimple.mockReturnValue(mockIterator as AssistantMessageEventStream);

      const onProgress = vi.fn();

      const result = await compactWithProgress(
        mockPreparation,
        mockModel,
        mockApiKey,
        customInstructions,
        undefined,
        onProgress,
      );

      expect(result.summary).toContain("Performance-focused summary");
      expect(mockStreamSimple).toHaveBeenCalledWith(
        mockModel,
        expect.objectContaining({
          messages: expect.arrayContaining([
            expect.objectContaining({
              content: expect.arrayContaining([
                expect.objectContaining({
                  text: expect.stringContaining(customInstructions),
                }),
              ]),
            }),
          ]),
        }),
        expect.any(Object),
      );
    });
  });
});
