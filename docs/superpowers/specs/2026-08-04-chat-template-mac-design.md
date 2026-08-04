# Chat Template for macOS — Design

**Date:** 2026-08-04
**Source:** https://github.com/EvanBacon/chat-template (Expo/React Native AI chat template)
**Goal:** Port the chat-template experience to a native SwiftUI macOS desktop app inside the Mference repo, as additive targets that do not touch existing Mference code.

## Scope

A ChatGPT-style desktop chat app with the template's feature set, adapted to macOS idioms:

- Collapsible sidebar with chat history grouped by recency (Starred / N days ago), star/rename/delete context menus, and a New Chat action — mirroring `chats.tsx` + `sidebar.web.tsx`, seeded with the template's `MOCK_CHATS`.
- Streaming assistant responses with throttled (~30 fps) UI updates via a streaming store, shimmer loading state before the first token, and a scroll-to-bottom button — mirroring `streaming-store.ts`, `conversation.tsx`, `streaming-message.tsx`.
- Markdown rendering: fenced code blocks (monospaced, background, copy button), GFM tables, headings, lists, block quotes, and inline formatting — mirroring the template's custom AST renderer.
- Glass prompt composer (`.ultraThinMaterial`, rounded) with a plus action, send button that turns into stop while generating, Return to send / Shift-Return for newline — mirroring `prompt-input.tsx` and the Liquid Glass composer.
- Model picker menu in the toolbar with an "extended thinking" toggle — mirroring `model-context.tsx` / `model-picker.tsx`.
- Automatic light/dark appearance using system semantic colors (stand-in for the template's OKLCH tokens).

Out of scope (template features that don't translate or are mock-only there): attachments screen, settings/profile screens, haptics, keyboard-controller. Chat history is in-memory per session, as in the template.

## Backend

The template ships mock streaming by default and an Anthropic API route behind it. The desktop port keeps the same shape:

- `ChatBackend` protocol: `stream(messages:model:) -> AsyncThrowingStream<String>`.
- `MockChatBackend`: port of the template's `MOCK_RESPONSES` with word-by-word delayed streaming.
- `OpenAICompatBackend`: SSE streaming against any OpenAI-compatible `/v1/chat/completions` (`stream: true`), default base URL `http://127.0.0.1:8080/v1` — Mference's local server. Models listed via `/v1/models` with a static fallback.
- Settings (Settings scene): base URL, optional API key (stored in the login keychain, not UserDefaults), mock-mode toggle. Mock is the default so the app works out of the box, exactly like the template. An invalid base URL with mock off surfaces an error instead of silently mocking.

## Architecture

Two new SwiftPM targets (plus tests), following the repo's Core/Presentation split:

- `ChatTemplateCore` (library, no dependencies): `ChatMessage`, `Chat`, `ChatStore` (@Observable; send/stream/stop/star/rename/delete, throttled streaming buffer), `ChatBackend` protocol + mock + OpenAI-compatible client, `SSELineParser`, `MarkdownBlockParser` (block-level: code fences, tables, headings, lists, quotes, paragraphs; inline via `AttributedString(markdown:)` at render time), mock seed data.
- `ChatTemplateMac` (executable): `ChatTemplateApp`, `RootView` (NavigationSplitView), `SidebarView`, `ChatView`, `MessageView`, `StreamingTextView`, `MarkdownView` + `CodeBlockView`, `PromptComposer`, `ModelPickerMenu`, `SettingsView`.
- `ChatTemplateCoreTests` (test target): SSE parsing, markdown block parsing, ChatStore streaming lifecycle with a scripted fake backend.

Files stay small and single-purpose (repo convention, 200–400 lines).

## Error handling

- Network/stream errors surface as an error banner above the composer (template shows error state from `useChat`); the partial response is kept.
- Stop cancels the task; partial text is committed to the message. Deleting the generating chat also cancels. The store tracks which chat is receiving the stream (`generatingChatID`) so switching chats mid-generation never renders streaming text into the wrong transcript.
- Non-2xx responses decode the server's error message when possible.

## Verification

- `Scripts/test.sh --filter ChatTemplateCoreTests` passes.
- `swift build --product ChatTemplateMac` succeeds; app launches and mock chat streams end-to-end.
