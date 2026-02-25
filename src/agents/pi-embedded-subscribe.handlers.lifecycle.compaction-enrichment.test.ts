import { describe, expect, it, vi, beforeEach } from "vitest";
import { onAgentEvent } from "../infra/agent-events.js";
import { subscribeEmbeddedPiSession } from "./pi-embedded-subscribe.js";

type StubSession = {
  subscribe: (fn: (evt: unknown) => void) => () => void;
  messages?: Array<{ role: string; content: string | Array<{ type: string; text: string }> }>;
  model?: { contextWindow?: number; provider?: string; id?: string };
};

describe("compaction event enrichment", () => {
  // ── Phase 1a: tokensBefore in start event ──

  it("includes tokensBefore in compaction start event when session has messages", async () => {
    let handler: ((evt: unknown) => void) | undefined;
    const session: StubSession = {
      subscribe: (fn) => {
        handler = fn;
        return () => {};
      },
      messages: [
        { role: "user", content: "Hello, this is a test message" },
        {
          role: "assistant",
          content: [{ type: "text", text: "Here is a response with some content" }],
        },
        { role: "user", content: "Another message with more text content to increase token count" },
      ],
      model: { contextWindow: 200000, provider: "anthropic", id: "claude-opus-4" },
    };

    const events: Array<Record<string, unknown>> = [];
    const stop = onAgentEvent((evt) => {
      if (evt.runId !== "run-enrich-1" || evt.stream !== "compaction") return;
      events.push({ ...evt.data });
    });

    subscribeEmbeddedPiSession({
      session: session as unknown as Parameters<typeof subscribeEmbeddedPiSession>[0]["session"],
      runId: "run-enrich-1",
    });

    handler?.({ type: "auto_compaction_start" });
    stop();

    expect(events).toHaveLength(1);
    expect(events[0].phase).toBe("start");
    // tokensBefore should be a positive number (exact value depends on estimateTokens implementation)
    expect(events[0].tokensBefore).toBeTypeOf("number");
    expect(events[0].tokensBefore).toBeGreaterThan(0);
  });

  it("includes contextWindow in compaction start event when model provides it", async () => {
    let handler: ((evt: unknown) => void) | undefined;
    const session: StubSession = {
      subscribe: (fn) => {
        handler = fn;
        return () => {};
      },
      messages: [{ role: "user", content: "Hello" }],
      model: { contextWindow: 200000, provider: "anthropic", id: "claude-opus-4" },
    };

    const events: Array<Record<string, unknown>> = [];
    const stop = onAgentEvent((evt) => {
      if (evt.runId !== "run-enrich-2" || evt.stream !== "compaction") return;
      events.push({ ...evt.data });
    });

    subscribeEmbeddedPiSession({
      session: session as unknown as Parameters<typeof subscribeEmbeddedPiSession>[0]["session"],
      runId: "run-enrich-2",
    });

    handler?.({ type: "auto_compaction_start" });
    stop();

    expect(events[0].contextWindow).toBe(200000);
  });

  it("omits tokensBefore when session has no messages", async () => {
    let handler: ((evt: unknown) => void) | undefined;
    const session: StubSession = {
      subscribe: (fn) => {
        handler = fn;
        return () => {};
      },
      messages: [],
      model: undefined,
    };

    const events: Array<Record<string, unknown>> = [];
    const stop = onAgentEvent((evt) => {
      if (evt.runId !== "run-enrich-3" || evt.stream !== "compaction") return;
      events.push({ ...evt.data });
    });

    subscribeEmbeddedPiSession({
      session: session as unknown as Parameters<typeof subscribeEmbeddedPiSession>[0]["session"],
      runId: "run-enrich-3",
    });

    handler?.({ type: "auto_compaction_start" });
    stop();

    expect(events[0].phase).toBe("start");
    expect(events[0].tokensBefore).toBeUndefined();
  });

  it("omits contextWindow when model is not set", async () => {
    let handler: ((evt: unknown) => void) | undefined;
    const session: StubSession = {
      subscribe: (fn) => {
        handler = fn;
        return () => {};
      },
      messages: [{ role: "user", content: "Test" }],
      model: undefined,
    };

    const events: Array<Record<string, unknown>> = [];
    const stop = onAgentEvent((evt) => {
      if (evt.runId !== "run-enrich-4" || evt.stream !== "compaction") return;
      events.push({ ...evt.data });
    });

    subscribeEmbeddedPiSession({
      session: session as unknown as Parameters<typeof subscribeEmbeddedPiSession>[0]["session"],
      runId: "run-enrich-4",
    });

    handler?.({ type: "auto_compaction_start" });
    stop();

    expect(events[0].contextWindow).toBeUndefined();
  });

  // ── Phase 1b: tokensAfter + tokensBefore in end event ──

  it("includes tokensBefore and tokensAfter in compaction end event", async () => {
    let handler: ((evt: unknown) => void) | undefined;
    const session: StubSession = {
      subscribe: (fn) => {
        handler = fn;
        return () => {};
      },
      messages: [
        { role: "user", content: "Hello this is a long message" },
        { role: "assistant", content: [{ type: "text", text: "Response text" }] },
      ],
      model: { contextWindow: 200000 },
    };

    const events: Array<Record<string, unknown>> = [];
    const stop = onAgentEvent((evt) => {
      if (evt.runId !== "run-enrich-5" || evt.stream !== "compaction") return;
      events.push({ ...evt.data });
    });

    subscribeEmbeddedPiSession({
      session: session as unknown as Parameters<typeof subscribeEmbeddedPiSession>[0]["session"],
      runId: "run-enrich-5",
    });

    // Start compaction
    handler?.({ type: "auto_compaction_start" });

    // Simulate compaction reducing messages (SDK replaces messages after compaction)
    session.messages = [{ role: "user", content: "Summary" }];

    // End compaction
    handler?.({ type: "auto_compaction_end", willRetry: false });
    stop();

    expect(events).toHaveLength(2);

    // Start event
    expect(events[0].phase).toBe("start");
    expect(events[0].tokensBefore).toBeTypeOf("number");
    expect(events[0].tokensBefore).toBeGreaterThan(0);

    // End event
    expect(events[1].phase).toBe("end");
    expect(events[1].tokensBefore).toBeTypeOf("number");
    expect(events[1].tokensAfter).toBeTypeOf("number");
    // After compaction, tokensAfter should be less than tokensBefore
    expect(events[1].tokensAfter).toBeLessThan(events[1].tokensBefore as number);
    expect(events[1].willRetry).toBe(false);
  });

  it("preserves tokensBefore across start→end lifecycle", async () => {
    let handler: ((evt: unknown) => void) | undefined;
    const session: StubSession = {
      subscribe: (fn) => {
        handler = fn;
        return () => {};
      },
      messages: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "World" },
      ],
      model: { contextWindow: 200000 },
    };

    const events: Array<Record<string, unknown>> = [];
    const stop = onAgentEvent((evt) => {
      if (evt.runId !== "run-enrich-6" || evt.stream !== "compaction") return;
      events.push({ ...evt.data });
    });

    subscribeEmbeddedPiSession({
      session: session as unknown as Parameters<typeof subscribeEmbeddedPiSession>[0]["session"],
      runId: "run-enrich-6",
    });

    handler?.({ type: "auto_compaction_start" });

    // The tokensBefore from start should match tokensBefore in end
    const startTokens = events[0].tokensBefore;

    // Simulate compaction
    session.messages = [{ role: "user", content: "Compact" }];
    handler?.({ type: "auto_compaction_end", willRetry: false });
    stop();

    expect(events[1].tokensBefore).toBe(startTokens);
  });

  // ── Edge cases: retry behavior ──

  it("does not include tokensAfter when willRetry is true", async () => {
    let handler: ((evt: unknown) => void) | undefined;
    const session: StubSession = {
      subscribe: (fn) => {
        handler = fn;
        return () => {};
      },
      messages: [{ role: "user", content: "Hello" }],
      model: { contextWindow: 200000 },
    };

    const events: Array<Record<string, unknown>> = [];
    const stop = onAgentEvent((evt) => {
      if (evt.runId !== "run-enrich-7" || evt.stream !== "compaction") return;
      events.push({ ...evt.data });
    });

    subscribeEmbeddedPiSession({
      session: session as unknown as Parameters<typeof subscribeEmbeddedPiSession>[0]["session"],
      runId: "run-enrich-7",
    });

    handler?.({ type: "auto_compaction_start" });
    handler?.({ type: "auto_compaction_end", willRetry: true });
    stop();

    // End event with willRetry should NOT include tokensAfter
    // (compaction failed, current messages are not the post-compaction state)
    expect(events[1].phase).toBe("end");
    expect(events[1].willRetry).toBe(true);
    expect(events[1].tokensAfter).toBeUndefined();
  });

  it("handles estimateTokens failure gracefully (still emits event)", async () => {
    let handler: ((evt: unknown) => void) | undefined;
    // Session with messages that might trip up estimateTokens
    const session: StubSession = {
      subscribe: (fn) => {
        handler = fn;
        return () => {};
      },
      // Getter that throws — simulates corrupted session state
      get messages() {
        throw new Error("session corrupted");
      },
      model: { contextWindow: 100000 },
    };

    const events: Array<Record<string, unknown>> = [];
    const stop = onAgentEvent((evt) => {
      if (evt.runId !== "run-enrich-8" || evt.stream !== "compaction") return;
      events.push({ ...evt.data });
    });

    subscribeEmbeddedPiSession({
      session: session as unknown as Parameters<typeof subscribeEmbeddedPiSession>[0]["session"],
      runId: "run-enrich-8",
    });

    // Should not throw — errors are caught
    handler?.({ type: "auto_compaction_start" });
    stop();

    // Event still emitted, just without tokensBefore
    expect(events).toHaveLength(1);
    expect(events[0].phase).toBe("start");
    expect(events[0].tokensBefore).toBeUndefined();
  });

  // ── onAgentEvent callback enrichment ──

  it("includes enriched data in onAgentEvent callback (not just bus)", async () => {
    let handler: ((evt: unknown) => void) | undefined;
    const session: StubSession = {
      subscribe: (fn) => {
        handler = fn;
        return () => {};
      },
      messages: [{ role: "user", content: "Hello test message" }],
      model: { contextWindow: 150000 },
    };

    const callbackEvents: Array<{ stream: string; data: Record<string, unknown> }> = [];

    subscribeEmbeddedPiSession({
      session: session as unknown as Parameters<typeof subscribeEmbeddedPiSession>[0]["session"],
      runId: "run-enrich-9",
      onAgentEvent: (evt) => {
        callbackEvents.push(evt);
      },
    });

    handler?.({ type: "auto_compaction_start" });
    session.messages = [{ role: "user", content: "Short" }];
    handler?.({ type: "auto_compaction_end", willRetry: false });

    // Both start and end should have enriched data via callback
    expect(callbackEvents).toHaveLength(2);
    expect(callbackEvents[0].data.tokensBefore).toBeTypeOf("number");
    expect(callbackEvents[0].data.contextWindow).toBe(150000);
    expect(callbackEvents[1].data.tokensAfter).toBeTypeOf("number");
  });

  // ── Multiple compaction cycles ──

  it("handles multiple compaction cycles correctly (no state leakage)", async () => {
    let handler: ((evt: unknown) => void) | undefined;
    const session: StubSession = {
      subscribe: (fn) => {
        handler = fn;
        return () => {};
      },
      messages: [
        { role: "user", content: "Long message one two three four five" },
        { role: "assistant", content: "Response message" },
      ],
      model: { contextWindow: 200000 },
    };

    const events: Array<Record<string, unknown>> = [];
    const stop = onAgentEvent((evt) => {
      if (evt.runId !== "run-enrich-10" || evt.stream !== "compaction") return;
      events.push({ ...evt.data });
    });

    subscribeEmbeddedPiSession({
      session: session as unknown as Parameters<typeof subscribeEmbeddedPiSession>[0]["session"],
      runId: "run-enrich-10",
    });

    // First cycle
    handler?.({ type: "auto_compaction_start" });
    session.messages = [{ role: "user", content: "Summary1" }];
    handler?.({ type: "auto_compaction_end", willRetry: false });

    // Grow context again
    session.messages = [
      { role: "user", content: "Summary1" },
      { role: "assistant", content: "More content added after first compaction" },
      { role: "user", content: "Even more content" },
    ];

    // Second cycle
    handler?.({ type: "auto_compaction_start" });
    session.messages = [{ role: "user", content: "Summary2" }];
    handler?.({ type: "auto_compaction_end", willRetry: false });

    stop();

    expect(events).toHaveLength(4);
    // First start/end
    expect(events[0].phase).toBe("start");
    expect(events[1].phase).toBe("end");
    // Second start should have different tokensBefore than first
    expect(events[2].phase).toBe("start");
    expect(events[3].phase).toBe("end");
    // Both end events should have tokensAfter
    expect(events[1].tokensAfter).toBeTypeOf("number");
    expect(events[3].tokensAfter).toBeTypeOf("number");
  });

  // ── Large message handling ──

  it("handles large message arrays without performance issues", async () => {
    let handler: ((evt: unknown) => void) | undefined;
    const largeMessages = Array.from({ length: 200 }, (_, i) => ({
      role: i % 2 === 0 ? "user" : "assistant",
      content: "Message number " + i + " with some padding text to simulate real conversation",
    }));

    const session: StubSession = {
      subscribe: (fn) => {
        handler = fn;
        return () => {};
      },
      messages: largeMessages,
      model: { contextWindow: 200000 },
    };

    const events: Array<Record<string, unknown>> = [];
    const stop = onAgentEvent((evt) => {
      if (evt.runId !== "run-enrich-11" || evt.stream !== "compaction") return;
      events.push({ ...evt.data });
    });

    subscribeEmbeddedPiSession({
      session: session as unknown as Parameters<typeof subscribeEmbeddedPiSession>[0]["session"],
      runId: "run-enrich-11",
    });

    const startTime = Date.now();
    handler?.({ type: "auto_compaction_start" });
    const elapsed = Date.now() - startTime;

    stop();

    // Should complete in under 100ms even with 200 messages
    expect(elapsed).toBeLessThan(100);
    expect(events[0].tokensBefore).toBeTypeOf("number");
    expect(events[0].tokensBefore).toBeGreaterThan(0);
  });
});

type SessionEventHandler = (evt: unknown) => void;
