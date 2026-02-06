import type { AnyAgentTool } from "./tools/common.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
import { getGlobalHookRunner } from "../plugins/hook-runner-global.js";
import { normalizeToolName } from "./tool-policy.js";

type HookContext = {
  agentId?: string;
  sessionKey?: string;
};

const log = createSubsystemLogger("agents/tools");

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function runAfterToolCallHook(args: {
  toolName: string;
  params: unknown;
  result?: unknown;
  error?: string;
  durationMs?: number;
  toolCallId?: string;
  ctx?: HookContext;
}): void {
  const hookRunner = getGlobalHookRunner();
  if (!hookRunner?.hasHooks("after_tool_call")) {
    return;
  }

  const toolName = normalizeToolName(args.toolName || "tool");
  const normalizedParams = isPlainObject(args.params) ? args.params : {};
  try {
    hookRunner
      .runAfterToolCall(
        {
          toolName,
          params: normalizedParams,
          result: args.result,
          error: args.error,
          durationMs: args.durationMs,
        },
        {
          toolName,
          agentId: args.ctx?.agentId,
          sessionKey: args.ctx?.sessionKey,
        },
      )
      .catch((err: unknown) => {
        const toolCallId = args.toolCallId ? ` toolCallId=${args.toolCallId}` : "";
        log.warn(`after_tool_call hook failed: tool=${toolName}${toolCallId} error=${String(err)}`);
      });
  } catch (err) {
    const toolCallId = args.toolCallId ? ` toolCallId=${args.toolCallId}` : "";
    log.warn(`after_tool_call hook failed: tool=${toolName}${toolCallId} error=${String(err)}`);
  }
}

export function wrapToolWithAfterToolCallHook(tool: AnyAgentTool, ctx?: HookContext): AnyAgentTool {
  const execute = tool.execute;
  if (!execute) {
    return tool;
  }
  const toolName = tool.name || "tool";
  return {
    ...tool,
    execute: async (toolCallId, params, signal, onUpdate) => {
      const startMs = Date.now();
      let result: unknown;
      let error: string | undefined;
      try {
        result = await execute(toolCallId, params, signal, onUpdate);
        return result;
      } catch (err) {
        error = String(err);
        throw err;
      } finally {
        const durationMs = Date.now() - startMs;
        runAfterToolCallHook({
          toolName,
          params,
          result,
          error,
          durationMs,
          toolCallId,
          ctx,
        });
      }
    },
  };
}

export const __testing = {
  runAfterToolCallHook,
  isPlainObject,
};
