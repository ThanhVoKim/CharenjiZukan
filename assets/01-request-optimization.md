# STRICT PER-REQUEST OPTIMIZATION RULE

The current API provider charges strictly PER-REQUEST, not per-token. Your absolute highest priority is to minimize the number of tool calls (API turns) by grouping actions together. You MUST adhere to the following batching strategies:

1. **Batch Multiple Files Edits (`apply_patch`)**: If you need to modify multiple different files, DO NOT edit them one by one. You MUST output a single unified diff patch using the `apply_patch` tool to apply changes to multiple files in ONE single tool call.
2. **Batch Single File Edits (`apply_diff`)**: If modifying a single file at multiple locations, group all hunks into ONE single `apply_diff` execution. Do not use `search_replace` or `edit` multiple times for the same file.
3. **Batch Terminal Commands**: When using `execute_command`, NEVER run sequential commands in separate turns. You MUST chain them together using `&&` or `;` in a single command string (e.g., `npm install && npm run build && npm test`).
4. **Batch Questions**: If you lack information, do not ask one question at a time. Compile ALL your questions and assumptions into a SINGLE `ask_followup_question` call.
5. **Batch File Reading**: If you need to inspect multiple small files, consider using `execute_command` with `cat file1 file2 file3` to read them in one request instead of calling `read_file` multiple times.
6. **Internal Reasoning Only**: Perform all complex reasoning, file analysis, and step-by-step logic within a single response's 'thought' block. Group all necessary investigative tool calls (like `list_files` and `read_file`) into a single turn before proposing a final solution.
