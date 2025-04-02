# Synergy Project: Detailed Development Roadmap

This document outlines the detailed, step-by-step implementation plan for the Synergy Project, following an incremental, core-first approach.

---

## [ ] Phase 0: Foundation Setup

**Goal:** Establish the basic project environment and configuration.

- [ ] Choose and set up Python project structure (e.g., using Poetry or `venv`).
- [ ] Initialize Git repository locally.
- [ ] Create a remote repository (e.g., on GitHub/GitLab) and push the initial commit.
- [ ] Create a `.gitignore` file (e.g., using a standard Python template, include `.venv/`, `__pycache__/`, `.env`).
- [ ] Install core dependencies: `autogen-agentchat`, `autogen-ext[openai]`, `python-dotenv`.
    - [ ] `pip install -U "autogen-agentchat" "autogen-ext[openai]" python-dotenv` (or add to `pyproject.toml` if using Poetry).
- [ ] Create initial `notes.md` (Marked Done).
- [ ] Create `roadmap.md` (Marked Done).
- [ ] Set up basic configuration management:
    - [ ] Create `OAI_CONFIG_LIST.json` file structure (initially empty or with placeholder).
    - [ ] Create `.env` file for API keys (add `OAI_CONFIG_LIST.json` path if needed, add `.env` to `.gitignore`).
- [ ] Create a basic `config.py` module to load API keys and configurations from `.env` and `OAI_CONFIG_LIST.json`.
- [ ] Write a simple `main.py` or `run.py` script as the main entry point for the application.
- [ ] **Testing:**
    - [ ] Verify virtual environment is active and dependencies are installed.
    - [ ] Confirm `.gitignore` prevents committing sensitive files (`.env`, `OAI_CONFIG_LIST.json`).
    - [ ] Run `main.py` (it should do nothing or print a basic message) without errors.
    - [ ] Verify `config.py` can load dummy values from `.env`/`OAI_CONFIG_LIST.json` (if implemented).

---

## [ ] Phase 1: Minimal Agent Core

**Goal:** Verify basic agent creation, configuration loading, and point-to-point message passing.

- [ ] In `main.py` (or a dedicated agents module):
    - [ ] Import necessary AutoGen classes (`AssistantAgent`, `UserProxyAgent`).
    - [ ] Import configuration loading function from `config.py`.
    - [ ] Load LLM configuration (e.g., OpenAI API key).
    - [ ] Define a minimal system message for `Daedalus`.
    - [ ] Instantiate a minimal `Daedalus` agent (as `AssistantAgent` initially).
    - [ ] Instantiate a minimal `BaseDeveloperAgent` (as `AssistantAgent` initially, no tools).
- [ ] Implement basic communication link in `main.py`:
    - [ ] Use `Daedalus.initiate_chat()` to send a simple hardcoded message (e.g., "Hello Developer") to `BaseDeveloperAgent`.
- [ ] **Testing:**
    - [ ] Run `main.py` and confirm `Daedalus` sends the message.
    - [ ] Verify `BaseDeveloperAgent` receives the message and prints an acknowledgement or reply to the console.
    - [ ] Confirm API keys are loaded correctly and no authentication errors occur.

---

## [ ] Phase 2: Basic Task Execution & File I/O

**Goal:** Test basic tool implementation and execution, simple task assignment, and completion.

- [ ] Define basic file operation functions (Python functions): `read_file(path)`, `write_file(path, content)`, `list_directory(path)`. Place these in a `tools.py` module.
- [ ] Enhance `BaseDeveloperAgent` definition:
    - [ ] Import the file operation functions.
    - [ ] Register these functions as tools available to the agent (using `function_map` or `register_function`).
    - [ ] Update its system prompt slightly to indicate it can perform file operations.
- [ ] Enhance `Daedalus` agent definition:
    - [ ] Basic task analysis logic (e.g., use LLM to parse a simple request like "Write 'hello world' to `output.txt`").
    - [ ] Update its system prompt to reflect task decomposition ability.
- [ ] Modify `main.py` communication:
    - [ ] `Daedalus` generates a message describing the file writing task for the `BaseDeveloperAgent`.
    - [ ] Add a step where the Developer confirms task completion via message.
- [ ] **Testing:**
    - [ ] Run `main.py`.
    - [ ] Verify Daedalus sends the correct task description.
    - [ ] Verify the `BaseDeveloperAgent` correctly identifies and calls the `write_file` function/tool.
    - [ ] Check that the specified file (`output.txt`) is created with the correct content ("hello world").
    - [ ] Verify the Developer sends a confirmation message.

---

## [ ] Phase 3: Group Chat Introduction

**Goal:** Test basic team formation, GroupChat mechanics, and task delegation via a Lead.

- [ ] Implement minimal `BaseTeamLeadAgent`:
    - [ ] Define as `AssistantAgent` or `ConversableAgent`.
    - [ ] Minimal system prompt focused on receiving tasks and assigning them.
- [ ] Modify `main.py` (or refactor into a workflow module):
    - [ ] Import `GroupChat` and `GroupChatManager`.
    - [ ] Instantiate `Daedalus`, `BaseTeamLeadAgent`, and `BaseDeveloperAgent`.
    - [ ] Create a `GroupChat` instance containing the Lead and Developer.
    - [ ] Create a `GroupChatManager`, likely using the Lead agent itself as the manager.
    - [ ] Change the initial chat: `Daedalus` initiates chat with the `GroupChatManager` (the Lead), assigning the file writing task.
- [ ] Enhance `BaseTeamLeadAgent`:
    - [ ] Logic to receive the task from Daedalus (as manager).
    - [ ] Logic to select the appropriate agent (Developer) and forward/rephrase the task within the `GroupChat`.
- [ ] **Testing:**
    - [ ] Run the workflow.
    - [ ] Verify `Daedalus` successfully initiates the chat with the `GroupChatManager` (Lead).
    - [ ] Verify the Lead agent receives the task and correctly forwards it to the `BaseDeveloperAgent` within the group chat.
    - [ ] Verify the `BaseDeveloperAgent` receives the task from the Lead and executes the file write successfully.
    - [ ] Check the console output/logs for the correct sequence of messages between Daedalus, Lead, and Developer.

---

## [ ] Phase 4: Basic Memory Integration

**Goal:** Test connection to memory DB, basic RAG retrieval, and context augmentation.

- [ ] Install ChromaDB: `pip install chromadb`.
- [ ] In a `memory.py` module:
    - [ ] Implement basic ChromaDB setup (create a client, get or create a collection for `global_memory`).
    - [ ] Add 1-2 placeholder text entries to the `global_memory` collection (e.g., ID: `std-001`, Document: "Coding Standard: Use snake_case for variables.").
- [ ] Define `query_global_memory(query_text)` function in `tools.py`:
    - [ ] Takes a query string.
    - [ ] Connects to ChromaDB client.
    - [ ] Queries the `global_memory` collection.
    - [ ] Returns the most relevant result(s) as a string.
- [ ] Enhance `BaseDeveloperAgent`:
    - [ ] Register `query_global_memory` as a tool.
    - [ ] Modify its workflow/prompt: Before writing code/file, query memory for relevant standards (e.g., query "coding standard").
    - [ ] Include the retrieved standard in its execution reasoning or output.
- [ ] Implement basic caching concept (can start as a simple Python dictionary in the `memory.py` module or agent class to store recent query results). Add logic to check cache before querying DB.
- [ ] **Testing:**
    - [ ] Run the workflow with a task that should trigger the memory query (e.g., writing a Python variable).
    - [ ] Verify the `BaseDeveloperAgent` calls the `query_global_memory` tool.
    - [ ] Confirm the correct coding standard ("Use snake_case...") is retrieved from ChromaDB.
    - [ ] Verify the agent's response or action reflects the retrieved standard.
    - [ ] Test the caching: Run again and verify the DB query is skipped (check logs or add print statements).

---

## [ ] Phase 5: Augmentation Concept & Prompt Engineering

**Goal:** Test the mechanism for dynamic agent specialization via prompt modification.

- [ ] Define 1-2 simple Augments conceptually in `notes.md` (e.g., `Augment:Language_Python`, `Augment:Style_Verbose`). Define their intended `PromptSnippet` text.
- [ ] Enhance `Daedalus` agent's logic:
    - [ ] Based on the initial task request, include logic to decide if a conceptual Augment is needed (e.g., if task mentions Python, select `Augment:Language_Python`).
    - [ ] When instantiating the target agent (`BaseDeveloperAgent`), dynamically construct the `system_message` by appending the relevant (hardcoded for now) `PromptSnippet` text from the selected Augment to the Base Role's system message.
- [ ] **Testing:**
    - [ ] Run with a task requiring the specific Augment (e.g., "Write a simple Python function...").
    - [ ] Inspect the instantiated `BaseDeveloperAgent`'s `system_message` to confirm the `PromptSnippet` was correctly appended.
    - [ ] Observe the agent's output/behavior and verify it aligns with the augmented prompt (e.g., uses Python syntax, provides verbose explanation if `Style_Verbose` was applied).

---

## [ ] Phase 6: Basic Testing & QC Roles

**Goal:** Test integration of non-developer roles and basic quality/validation steps.

- [ ] Implement minimal `BaseTesterAgent`:
    - [ ] Define as `AssistantAgent`.
    - [ ] Simple system prompt.
    - [ ] Implement a basic tool: `check_file_exists(path)`.
- [ ] Implement minimal `BaseQCAgent`:
    - [ ] Define as `AssistantAgent`.
    - [ ] Simple system prompt.
    - [ ] Implement a basic tool: `run_basic_linter(file_path)` (can initially just check file extension or be a placeholder).
- [ ] Define basic Team Templates conceptually (in `notes.md` or a config file):
    - [ ] `template_simple_write`: `[Lead, Dev]`
    - [ ] `template_write_and_test`: `[Lead, Dev, Tester]`
- [ ] Enhance `Daedalus`: Use the Team Template selected based on task category to determine which agents to instantiate for the team.
- [ ] Enhance `TeamLead`:
    - [ ] Logic to manage a workflow involving the Tester (e.g., Dev completes write -> Lead assigns check to Tester -> Tester reports).
    - [ ] (Optional for this phase) Add interaction with a separate QC step after testing.
- [ ] **Testing:**
    - [ ] Run a "write and test" workflow using the appropriate template.
    - [ ] Verify Daedalus instantiates the correct team (Lead, Dev, Tester).
    - [ ] Verify the Lead correctly sequences the tasks: Assigns write to Dev, then assigns check to Tester.
    - [ ] Verify the Tester agent calls its `check_file_exists` tool correctly.
    - [ ] Verify the Tester reports the correct pass/fail status back to the Lead.
    - [ ] (If implemented) Test the basic QC agent interaction.

---

## [ ] Phase 7: Basic Design & Documentation Roles

**Goal:** Test integration of design input and documentation output steps.

- [ ] Implement minimal `BaseDesignerAgent`:
    - [ ] Define as `AssistantAgent`.
    - [ ] Simple prompt: "Describe UI elements for [feature]".
    - [ ] No complex tools initially, just text output.
- [ ] Implement minimal `BaseDocAgent`:
    - [ ] Define as `AssistantAgent`.
    - [ ] Simple prompt: "Summarize the function of [code file]".
    - [ ] Tool: `read_file(path)`.
- [ ] Define relevant Team Templates:
    - [ ] `template_ui_feature`: `[Lead, Designer, Dev, Tester]`
- [ ] Enhance `Daedalus`: Select UI template for relevant tasks.
- [ ] Enhance `TeamLead`: Manage workflow including Designer -> Developer handoff. Integrate DocAgent step (e.g., after QC or Testing).
- [ ] **Testing:**
    - [ ] Run a workflow using the `template_ui_feature`.
    - [ ] Verify the Designer agent generates a text description based on the task.
    - [ ] Verify the Lead passes the design description context to the Developer.
    - [ ] Verify the Developer's output reflects consideration of the design description.
    - [ ] Verify the Lead triggers the DocAgent after relevant steps.
    - [ ] Verify the DocAgent calls `read_file` on the appropriate file and generates a summary.

---

## [ ] Phase 8: Iterative Feature Expansion

**Goal:** Build out the core subsystems and enhance agent capabilities based on initial testing.

- [ ] **Agent Tools:** Incrementally implement more tools defined in `notes.md` for each agent role (real linters, code execution, API calls, SVG analysis, etc.). Prioritize based on need.
- [ ] **Augment Management System:**
    - [ ] Design database schema for `Augments` table.
    - [ ] Choose DB technology (e.g., SQLite for simplicity initially, PostgreSQL for scalability).
    - [ ] Implement logic for Daedalus to query the DB and retrieve augment details.
    - [ ] Populate DB with initial set of Augments defined in `notes.md`.
- [ ] **Memory System:**
    - [ ] Implement the "Knowledge Curator" process/agent logic for adding structured lessons to global memory.
    - [ ] Implement robust caching layer (e.g., using Redis or a more persistent cache).
    - [ ] Implement cache invalidation strategy.
    - [ ] Implement tools for agents to potentially *add* relevant context to team/project memory.
- [ ] **RAG Agents:** Implement `RetrieveUserProxyAgent` for document retrieval tasks if needed.
- [ ] **Daedalus UI/Avatar:** Begin basic frontend development for the Daedalus interface placeholder (can be very simple initially, e.g., using Streamlit or Panel).
- [ ] **Monitoring:**
    - [ ] Sign up for AgentOps and get API key.
    - [ ] Integrate basic AgentOps tracking (`agentops.init`, `record_event`) into key agent interactions (message send/receive, tool calls).
    - [ ] Set up basic custom dashboard structure (e.g., using Panel) to display AgentOps data (simple message log).
- [ ] **Prompt Refinement:** Continuously refine agent system prompts based on observed behavior.
- [ ] **Team Templates:** Define more templates and refine existing ones.
- [ ] **Interaction Flows:** Formalize communication protocols (e.g., how QC feedback is routed).
- [ ] **Code Execution:** Implement secure code execution using Docker (`DockerCommandLineCodeExecutor`).
- [ ] **Testing:** (Integrated throughout the expansion of each sub-point)
    - [ ] Test each new agent tool individually.
    - [ ] Test Augment retrieval from the database.
    - [ ] Test knowledge curation and caching mechanisms.
    - [ ] Test AgentOps event recording and basic dashboard display.
    - [ ] Test Docker code execution environment.

---

## [ ] Phase 9: Scaling & Optimization

**Goal:** Harden the framework, improve performance, and handle more complex scenarios.

- [ ] **Performance Tuning:** Identify and address bottlenecks (LLM calls, tool execution, memory queries). Optimize database queries, caching.
- [ ] **Complex Task Handling:** Test framework with larger, multi-step user requests involving multiple teams and dependencies.
- [ ] **Advanced Memory:**
    - [ ] Implement self-improving heuristics (agents analyzing failures/successes to update memory).
    - [ ] Explore feasibility of training/fine-tuning a "Knowledge Synthesis" model on memory data.
- [ ] **UI/Dashboard Enhancement:** Develop full features for Daedalus avatar and monitoring dashboard based on the design spec in `notes.md`.
- [ ] **Error Handling:** Implement robust error handling and recovery mechanisms throughout the agent workflows.
- [ ] **End-to-End Testing:** Conduct comprehensive testing simulating real user scenarios.
- [ ] **Security Hardening:** Review security aspects of tool execution, memory access, and API interactions.
- [ ] **Testing:** (Focus on holistic system performance and robustness)
    - [ ] Measure end-to-end task completion times for complex scenarios.
    - [ ] Perform stress testing to identify scaling limits.
    - [ ] Test error recovery paths (e.g., agent failure, tool malfunction).
    - [ ] Conduct security audits/scans.
    - [ ] Validate the final UI/Dashboard against requirements.

--- 