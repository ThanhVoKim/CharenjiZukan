# Project Memory Rule

To ensure other Agents can seamlessly continue the work, the current Agent MUST adhere to the following:

1. **Maintain `JOURNAL.md`**: All major architectural changes, key decisions, or workflow modifications must be logged in the `JOURNAL.md` file located in the `logs` directory.
2. **Provide Current Context**: Before concluding a task, the Agent must summarize the current state of the project, including:
   - What has been completed.
   - Outstanding issues (Pending tasks).
   - Proposed next steps.
3. **Data Flow**: Always refer to the diagrams or flow documentation in the `docs/` directory to ensure new code does not break the overall system logic.
