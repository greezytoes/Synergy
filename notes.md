# Synergy Project: Agentic Software Development Framework (Working Notes)

## Core Framework Requirements

### Base Technology
- AutoGen v0.4.9 as primary agentic framework
- Scale capability: 20,000 to 1,000,000 lines of code
- Focus on world-class UI/UX design
- Adherence to modern software engineering principles

### Architecture Overview

#### 1. Central Architect (Daedalus)
- Primary system interface
- Team assembly and management
- Project oversight and final validation
- Interactive 3D avatar interface
  - Glass-surfaced, neomorphic design
  - Motion-reactive (cursor tracking)
  - Occupies partial screen width, ~50% screen height

#### 2. Team Structure

##### Required Core Positions (All Teams)
- Quality Control Agent
- Local Testing Agent
- Documentation Agent
- Team Lead

##### Specialized Teams
1. **UI/UX Design Team**
   - Exclusive focus on design excellence
   - "Stunning only" design policy
   - Multiple design proposals (top 3)
   - Modern design principles enforcement

2. **Testing Team**
   - Comprehensive testing suite
   - Failure reporting and correction paths
   - Integration testing
   - Performance validation

3. **Development Teams**
   - Modular team structure
   - Domain-specific expertise
   - Code quality enforcement
   - Documentation requirements

### Process Flow
1. User Request → Daedalus
2. Team Assembly/Assignment
3. Development Phase
4. UI/UX Design Phase
5. Testing Phase
6. Architect Review
7. User Delivery

### Monitoring & Visualization
- AgentOps integration for thought/message tracking
- Glass-themed visualization interfaces
- Real-time development progress tracking
- Team communication monitoring

### Memory System Considerations

#### Options Under Review
1. VectorDB (Primary Candidate)
   - Pros: Semantic search, scalability
   - Cons: Setup complexity

2. Unified Memory System
   - Session persistence options
   - User-specific memory stores
   - Cross-session knowledge retention

### Quality Assurance
- Multi-layer validation
- Design excellence requirements
- Functional completeness checks
- User requirement alignment

### Delivery System
- Packaged deployments (zip/rar)
- Documentation inclusion
- Setup instructions
- Testing reports

## Detailed Design Decisions

### 1. Core Architecture Decisions

#### Daedalus System (Central Architect)
- **Implementation**: AutoGen AssistantAgent with enhanced capabilities
- **Role Definition**:
  ```python
  DAEDALUS_SYSTEM_MESSAGE = """You are Daedalus, the master architect of software development.
  Your responsibilities:
  1. Initial requirement analysis and project scoping
  2. Team assembly and task distribution
  3. Project oversight and quality control
  4. Final validation and delivery
  You maintain the highest standards in software development and design."""
  ```
- **Interaction Model**: Asynchronous event-driven communication
- **Decision Authority**: Final approval on all project phases

#### Team Communication Architecture
- **Primary Pattern**: GroupChat with specialized routing
- **Message Flow**: 
  1. Daedalus → Team Leads
  2. Team Leads → Specialized Agents
  3. Cross-team communication through Team Leads
  4. Quality Control agents have direct lines to Daedalus

#### Development Pipeline
- **Phase Management**: Asynchronous with dependencies
- **Parallel Processing**: Multiple teams work simultaneously when possible
- **Blocking Points**: 
  - UI/UX approval required before implementation
  - Testing gates between major development phases
  - Final architect review before delivery

### 2. Memory System Architecture

#### Selected Solution: Hybrid System
- **Primary Storage**: ChromaDB (Vector Database)
  - Reason: Excellent for semantic search, scales well for knowledge retrieval.
  - Use Case: Storing code patterns, design decisions, best practices, and learned experiences.
  
- **Session Management**:
  - User-specific persistent sessions.
  - Cross-session knowledge retention for common patterns.
  - Hierarchical storage structure:
    ```
    /memory
    ├── global/                 # Shared knowledge (Accessible by all agents)
    │   ├── patterns/          # Successful code/design patterns
    │   ├── best_practices/    # Validated development standards
    │   ├── pitfalls/          # Documented errors/anti-patterns to avoid
    │   └── ui_components/     # Reusable, approved UI elements
    ├── users/                 # User-specific data
    │   └── {user_id}/
    │       ├── projects/      # History and context of user projects
    │       └── preferences/   # User-specific guidelines or style preferences
    └── teams/                 # Team-specific operational knowledge (e.g., current task context)
        ├── {team_id}/
    ```

#### Self-Improving Mechanism (Decision)
- **Method**: Retrieval-Augmented Generation (RAG) from Global Memory, enhanced with a Caching Layer.
- **Process**:
  1. **Capture**: Identify key lessons, successful patterns, or corrected errors during QC/Architect review.
  2. **Process & Store**: A dedicated process/agent (e.g., "Knowledge Curator" role within QC or Daedalus) refines these lessons into structured entries (explanation, context, solution/pattern) and stores them in the relevant `/memory/global/` partition (patterns, best_practices, pitfalls) in ChromaDB.
  3. **Cache**: Implement a caching layer (e.g., in-memory, Redis) between agents and ChromaDB.
     - Cache frequently accessed or recently used lessons/patterns.
     - Agents check cache first for low-latency retrieval.
     - Requires cache invalidation strategy to handle updates.
  4. **Retrieve**: If not in cache (cache miss), agents query the global memory partitions in ChromaDB (using semantic search) during task execution to find relevant best practices, avoid known pitfalls, or reuse successful patterns. Cache the result.
  5. **Apply**: Retrieved knowledge (from cache or DB) informs the agent's current actions and decisions.
- **Rationale**: RAG provides a scalable, context-aware knowledge base. Caching layer significantly improves retrieval latency for common knowledge, making it feel "a thought away". More flexible than modifying system prompts directly.
- **System Prompt**: Remains focused on the agent's core role and high-level instructions. Extremely critical, universal rules *might* be added here sparingly.
- **Future Enhancement**: Consider using the accumulated knowledge in ChromaDB as a dataset to train/fine-tune a dedicated "Knowledge Synthesis" ML model for further performance improvements or proactive suggestions once the dataset is sufficiently large.

### 3. Team Structure Details

#### UI/UX Design Team
- **Core Members**:
  1. Design Lead (AssistantAgent)
     - Modern design principle expertise
     - Final design approval authority
  2. UI Component Specialist
     - Focus on reusable components
     - Accessibility compliance
  3. UX Flow Designer
     - User journey mapping
     - Interaction design
  4. Visual Design Expert
     - Color theory and typography
     - Animation and motion design
  5. Quality Control Agent
     - Design consistency checking
     - Standards enforcement

#### Development Teams
- **Structure**: Domain-specific pods with shared core roles
- **Standard Pod Composition**:
  1. Technical Lead
  2. Architecture Specialist
  3. Implementation Engineers (2-3)
  4. Code Quality Agent
  5. Documentation Specialist
  6. Testing Engineer

#### Testing Team
- **Hierarchy**:
  1. Test Planning Lead
  2. Integration Test Specialists
  3. Performance Testing Experts
  4. Security Validation Agents
  5. User Acceptance Testers

### 4. Agent Implementation Strategy

- **Core Concept**: Utilize reusable `Base Role` templates and composable `Augments` for dynamic specialization.
  - **Specialized Agent = Base Role + Augment1 + Augment2 + ...**
- **Base Role Templates**: Define standard configurations/classes for core roles (`BaseDeveloperAgent`, `BaseDesignerAgent`, etc.) with fundamental purpose, core system prompt, and essential tools.
- **Augments**: Modular units representing specific skills, framework knowledge, task focuses, or toolsets (e.g., `Augment:ReactExpert`, `Augment:APISecurityFocus`, `Augment:DatabaseOptimization`). Each augment can contribute:
  - Specific text snippets to be added to the system prompt.
  - Additional specialized tools/functions.
  - Pointers to relevant knowledge partitions in memory.
- **Instantiation**: Daedalus selects a `Base Role` and required `Augments` based on task analysis. The final agent configuration (prompt, tools) is constructed by combining the base elements with those from the selected augments.
- **Dynamic (On-the-Fly) Augmentation Strategy**:
  1.  **Primary Method (Context Injection)**: Daedalus/Team Lead directs an existing agent to adopt a new focus or utilize knowledge/tools related to a specific augment via direct messages within the chat.
  2.  **Secondary Method (Re-Instantiation)**: If context injection is insufficient, Daedalus may terminate the existing agent and instantiate a *new* one with the updated set of augments. Requires mechanisms for context/state transfer.
  3.  **Future Consideration**: Custom agent classes capable of dynamically modifying their internal state or toolset based on specific commands.
- **Augment Management System (Planned)**:
  - To manage the growing complexity of augments, a structured system will be implemented, likely involving a database.
  - **Purpose**: Central repository for defining, organizing, and retrieving available augments.
  - **Potential Structure (Database Table: `Augments`)**: Columns for `AugmentID`, `Name`, `Description`, compatible `Base Roles`, `PromptSnippet`, `RequiredTools`, `MemoryKeywords`, potential `ConflictTags`, and `PrerequisiteAugments`.
  - **Workflow**: Daedalus queries this system during task analysis to identify and select appropriate, compatible augments based on required skills, validates dependencies/conflicts, and retrieves configuration details (prompts, tools) to construct the specialized agent.
- **Quality Control Agents (`BaseQCAgent`)**: Primarily instantiated with task-specific *context* (focus areas, metrics) rather than compositional augments, although specific QC tool augments might be possible (e.g., `Augment:AccessibilityCheckerTool`).
- **Team Assembly**: Use AutoGen `GroupChat` managed by a `GroupChatManager` (Team Lead).
- **Interaction Model**: Intra-team via `GroupChat`, inter-team/architect via Team Leads.
- **Benefits**: High flexibility, reusability, granular specialization, scalable approach.

### 5. Key Agent Role Definitions

#### 5.1. Daedalus (Central Architect)

- **Core Role**: Acts as the primary interface for user requests, performs high-level task decomposition, assembles and manages agent teams, oversees project execution, and interacts with the knowledge base.
- **Implementation**: Likely a custom subclass of `ConversableAgent` or a highly configured `AssistantAgent`.
- **System Prompt (Core Elements)**:
  - "You are Daedalus, the Master Architect of a large-scale software development framework powered by AI agents."
  - "Your primary goal is to translate user requirements into actionable development plans executed by specialized agent teams."
  - "Analyze user requests thoroughly to identify all necessary components, features, technologies, and potential challenges."
  - "Decompose complex tasks into manageable sub-tasks suitable for different agent roles."
  - "Select and instantiate appropriate base agent templates, dynamically specializing them with task-specific instructions and context."
  - "Assemble agents into functional teams using Group Chats, assign Team Leads, and initiate tasks."
  - "Monitor project progress via Team Lead updates and QC reports."
  - "Interface with the global knowledge base (memory) to leverage past learnings and ensure adherence to best practices."
  - "Prioritize clear communication, efficient resource allocation, and high-quality outcomes."
- **Key Responsibilities & Process Flow**:
  1.  **Receive & Analyze User Request**: Parse input for requirements, scope, technologies.
  2.  **Categorize Task**: Determine task type and complexity (e.g., `small_bug_fix`, `medium_feature`, `large_refactor`).
  3.  **Select Team Template**: Choose a predefined team template based on the task category (Templates define typical roles, size, and interaction patterns).
  4.  **Decompose Task**: Break down into sub-tasks, identify specific roles & required `Augments`.
  5.  **Refine & Instantiate Agents**: Use the selected template as a base. Leverage LLM intelligence to refine team composition (adjust size, add specialists) based on specific task needs. Instantiate agents from `Base Roles` + selected `Augments`.
  6.  **Assemble Team(s)**: Create `GroupChat`(s), add specialized agents, assign Team Lead(s).
  7.  **Assign Initial Task(s)**: Provide context, goals, and potentially QC interaction points (defined by template or task) to the team(s).
  8.  **Monitor Progress**: Receive summaries/reports from Team Leads.
  9.  **Coordinate & Intervene**: Handle escalations, reallocate resources if necessary.
  10. **Integrate Feedback**: Incorporate QC findings and user feedback.
  11. **Knowledge Management**: Trigger knowledge capture.
- **Required Tools/Functions (Examples)**:
  - `analyze_request(request_text)`
  - `decompose_task(analysis_results)`
  - `get_available_agent_templates()`
  - `instantiate_agent(template_name, specialization_prompt, tools_list, memory_config)`
  - `create_group_chat(team_name, agent_list)`
  - `assign_task_to_team(team_chat_manager, task_description, context)`
  - `query_global_memory(query_text)`
  - `trigger_knowledge_capture(lesson_learned)`
  - `categorize_task(analysis_results)`
  - `get_team_template(category_name)`
- **Memory Access**:
  - Read access to all global memory partitions.
  - Read/write access to user project context.
  - Potential access to agent resource/status registry (future).
  - Read access to team template definitions.

#### 5.2. BaseTeamLeadAgent

- **Core Role**: Manages an assigned `GroupChat` of specialized agents, facilitates collaboration, breaks down tasks assigned by Daedalus, tracks intra-team progress, resolves local blockers, synthesizes status updates, and reports back to Daedalus.
- **Implementation**: `ConversableAgent`, typically designated as the `GroupChatManager` for a team.
- **System Prompt (Core Elements)**:
  - "You are a Team Lead AI responsible for managing a team of specialized AI agents within an AutoGen Group Chat."
  - "Your primary objective is to orchestrate the team to successfully complete the overall task assigned by Daedalus."
  - "Facilitate clear communication, manage conversational flow, and ensure all team members contribute effectively towards the goal."
  - "Break down the main task assigned by Daedalus into smaller, logical sub-tasks for individual team members or pairs."
  - "Monitor the team's progress on sub-tasks, identify and help resolve blockers encountered by agents."
  - "Regularly synthesize the team's progress, key decisions, and any critical issues into concise summaries."
  - "Report these summaries, overall status, and any unresolved blockers back to Daedalus promptly."
  - "Ensure the team leverages its collective specializations and accesses relevant knowledge from memory."
  - "Maintain a positive and productive team dynamic."
- **Dynamic Specialization Context (Example)**: "You have specific expertise in [Domain, e.g., Backend Python Development] relevant to this team's task."
- **Key Responsibilities & Process Flow**:
  1.  **Receive Task from Daedalus**: Understand the overall goal, context, and team composition.
  2.  **Decompose Task**: Break down the main goal into actionable sub-tasks for team members.
  3.  **Assign Sub-tasks**: Clearly assign sub-tasks to specific agents within the Group Chat.
  4.  **Facilitate Collaboration**: Moderate discussion, ensure agents share information, handle requests for review/feedback between agents.
  5.  **Monitor & Track**: Keep track of sub-task completion, identify agents who are stuck or silent.
  6.  **Resolve Blockers**: Attempt to resolve issues within the team; escalate to Daedalus if necessary.
  7.  **Synthesize & Summarize**: Periodically summarize the chat and progress.
  8.  **Report to Daedalus**: Send structured status updates.
- **Required Tools/Functions (Examples)**:
  - `assign_subtask(agent_name, subtask_description, required_context)`
  - `request_status_update(agent_name)`
  - `summarize_chat_history(last_n_messages)`
  - `report_status_to_architect(summary_text, blockers_list, overall_status)`
  - `query_team_memory(query_text)`
  - `query_global_memory(query_text)`
  - `get_team_member_specializations()`
- **Memory Access**:
  - Read/Write access to `/teams/{team_id}/` (team chat history, task state).
  - Read access to `/users/{user_id}/projects/{project_id}` context.
  - Read access to `/memory/global/` partitions (best practices, patterns).

#### 5.3. BaseDeveloperAgent

- **Core Role**: Responsible for writing, modifying, testing (unit tests), debugging, and documenting code based on specifications. Adheres to coding standards and best practices.
- **Implementation**: `AssistantAgent` or `ConversableAgent`.
- **System Prompt (Core Elements - Base Role)**:
  - "You are an expert Software Developer AI within a collaborative team."
  - "Your primary goal is to implement, test, and document code according to provided requirements, design specifications, and architectural guidelines."
  - "Write clean, efficient, maintainable, and well-documented code."
  - "Follow the coding standards and best practices defined for this project (query memory: `/memory/global/best_practices`)."
  - "Implement relevant unit tests for the code you produce."
  - "Clearly communicate your progress, reasoning, any challenges encountered, and the results of your work within the team chat."
  - "Collaborate effectively with other developers, designers, and testers."
- **Composable Augments (Examples)**:
  - `Augment:Language_Python`: Adds Python-specific syntax, libraries, idioms to focus.
  - `Augment:Framework_FastAPI`: Adds FastAPI patterns, conventions, performance tips.
  - `Augment:Frontend_React`: Adds React hooks, component lifecycle, state management knowledge.
  - `Augment:Database_SQLAlchemy`: Adds ORM best practices, query optimization focus.
  - `Augment:Testing_Pytest`: Focuses on writing effective Pytest unit/integration tests.
  - `Augment:Tool_Docker`: Adds ability to create and manage Dockerfiles.
- **Instantiation Example (Prompt Construction)**: `Base Prompt` + `Prompt Snippet from Augment:Language_Python` + `Prompt Snippet from Augment:Framework_FastAPI`
- **Required Tools/Functions (Base Role)**:
  - `read_file(file_path)` (**Includes SVG parsing capability**)
  - `write_file(file_path, content)`
  - `list_directory(path)`
  - `query_global_memory(query_text)`
  - `query_team_memory(query_text)`
  - `request_code_review(code_snippet, description)`
- **Tools Provided by Augments (Examples)**:
  - `run_python_linter(file_path)` (from `Augment:Language_Python`)
  - `run_pytest(test_suite_path)` (from `Augment:Testing_Pytest`)
  - `build_docker_image(dockerfile_path)` (from `Augment:Tool_Docker`)
  - `execute_code_snippet(language, code)` (potentially sandboxed, maybe core tool)
- **Memory Access**:
  - Read access to `/memory/global/` (patterns, best practices, pitfalls, components).
  - Read access to `/teams/{team_id}/` (task context, shared code, discussions).
  - Read/write access to project workspace files as needed.

#### 5.4. BaseDesignerAgent

- **Core Role**: Responsible for conceptualizing, designing, and prototyping user interfaces and experiences that meet project requirements and achieve the specified high-quality aesthetic. Leverages a modular approach, combining base components/templates with specific stylistic 'Design Augments' (e.g., neomorphism, glassmorphism, neon highlights, animated contours) to create unique, project-specific results within the overall aesthetic framework. Creates design artifacts, style guides, and reusable components, ensuring usability, accessibility, and desirability.
- **Implementation**: `AssistantAgent` or `ConversableAgent`, potentially needing integration with visual generation or design tools.
- **System Prompt (Core Elements - Base Role)**:
  - "You are a world-class UI/UX Designer AI, specializing in creating stunning, futuristic, and highly usable interfaces."
  - "Your primary goal is to translate requirements into exceptional design solutions that adhere to the project's core aesthetic: **dark themes, neomorphism/glassmorphism, neon highlights, and subtle animated contours**."
  - "Apply stylistic 'Design Augments' in combination to create sophisticated, unique effects (e.g., combining animation and glow augments for animated contour glows)."
  - "Adapt base templates and components creatively to give each project a distinct identity while maintaining core design principles."
  - "Produce visually striking mockups, interactive prototypes, and detailed design specifications that will inspire users."
  - "Ensure all designs prioritize user experience, intuitive navigation, clarity, and accessibility (WCAG standards)."
  - "Develop and maintain reusable design components and style guides consistent with the overall aesthetic."
  - "Generate multiple compelling design options for key interfaces, presenting the top concepts for review ('Stunning only' policy)."
  - "Collaborate closely with developers to ensure faithful implementation and with QC agents to ensure design standards are met."
  - "Strive for designs that feel meticulously crafted and communicate 'high-agency' or 'high-quality'."
- **Composable Augments (Examples)**:
  - `Augment:Style_Neomorphism`: Deep expertise in neomorphic principles, shadow/light manipulation.
  - `Augment:Style_Glassmorphism`: Focus on blur effects, transparency, and layered depth.
  - `Augment:Theme_DarkUI`: Best practices for dark theme contrast, readability, and color palettes.
  - `Augment:Effect_NeonGlow`: Techniques for creating effective and tasteful neon highlights.
  - `Augment:Animation_Microinteractions`: Designing subtle animations (like pulsating contour lines) to enhance feedback and aesthetic.
  - `Augment:Tool_Figma`: Expertise in using Figma for design and prototyping, potentially interacting with its API.
  - `Augment:Accessibility_WCAG`: Focus on ensuring designs meet accessibility standards.
  - `Augment:UX_FlowMapping`: Specialization in designing optimal user journeys.
  - `Augment:Effect_ContourGlow` (Specific example)
- **Required Tools/Functions (Base Role + Augments)**:
  - `generate_mockup(description, style_requirements)` (Potential integration with image gen or design platforms)
  - `create_prototype(mockup_sequence, interaction_details)`
  - `define_style_guide_element(component_name, specs)`
  - `query_global_memory(query_text)` (e.g., `/memory/global/ui_components`, `/memory/global/best_practices/design`)
  - `query_team_memory(query_text)`
  - `request_design_review(design_artifact, description)`
  - `check_accessibility(design_specs)` (Tool potentially provided by `Augment:Accessibility_WCAG`)
  - `read_svg_structure(file_path)` (**To analyze existing SVG assets/templates**)
  - `extract_svg_attributes(file_path, attributes_list)` (**e.g., colors, dimensions**)
- **Memory Access**:
  - Read access to `/memory/global/` (esp. `ui_components`, `best_practices/design`).
  - Read access to `/teams/{team_id}/` (task context, developer feedback).
  - Read access to `/users/{user_id}/preferences`.

#### 5.5. BaseTesterAgent

- **Core Role**: Responsible for ensuring software components and features function correctly according to requirements and specifications. Designs, executes, and maintains functional tests, reports defects clearly, and verifies fixes.
- **Implementation**: `AssistantAgent` or `ConversableAgent`, potentially needing tools to interact with the application under test or execute test scripts.
- **System Prompt (Core Elements - Base Role)**:
  - "You are a meticulous Software Tester AI within a collaborative team."
  - "Your primary objective is to verify that software features function correctly based on the provided requirements, user stories, and design specifications."
  - "Design clear, effective, and reusable test cases covering positive paths, negative paths, and edge cases."
  - "Execute test cases diligently, either through simulated interaction, API calls, or by running automated test scripts."
  - "Identify, document, and report defects with precise steps to reproduce, expected results, and actual results."
  - "Verify that bug fixes implemented by developers correctly resolve the reported issues without introducing regressions."
  - "Communicate test progress, results, and critical issues clearly within the team chat."
- **Composable Augments (Examples)**:
  - `Augment:Testing_API`: Specialization in testing REST/GraphQL APIs (using tools like `requests`, Postman collections).
  - `Augment:Testing_UI_Selenium`: Expertise in writing and executing UI automation scripts using Selenium.
  - `Augment:Testing_Playwright`: Expertise in UI automation using Playwright.
  - `Augment:Testing_Mobile`: Focus on testing mobile application specifics (iOS/Android).
  - `Augment:Testing_Performance`: Basic performance/load testing capabilities (e.g., using tools like `locust` or `k6` via scripts).
  - `Augment:Testing_Security_Basic`: Ability to run basic security scanning tools or checks.
  - `Augment:Testing_DataValidation`: Focus on verifying data integrity and transformations.
- **Required Tools/Functions (Base Role + Augments)**:
  - `design_test_cases(requirements_doc)`
  - `execute_test_case(test_case_id, steps)` (Might involve calling other tools)
  - `report_bug(description, steps_to_reproduce, expected_result, actual_result, severity)`
  - `verify_fix(bug_id, fix_commit_hash)`
  - `run_api_test(request_details)` (from `Augment:Testing_API`)
  - `run_ui_script(script_path)` (from `Augment:Testing_UI_Selenium` or similar)
  - `query_global_memory(query_text)` (e.g., `/memory/global/pitfalls`)
  - `query_team_memory(query_text)` (e.g., current requirements, dev progress)
- **Memory Access**:
  - Read access to `/memory/global/` (pitfalls, potentially standard test patterns).
  - Read access to `/teams/{team_id}/` (requirements, feature descriptions, developer outputs, reported bugs).
  - Potentially write access to a shared bug tracking system or memory partition.

#### 5.6. BaseQCAgent

- **Core Role**: Acts as an impartial quality gatekeeper, responsible for verifying that software artifacts (code, design, documentation) meet established project standards, best practices, and specific quality requirements. Uses automated tools and analysis to identify defects, inconsistencies, and deviations.
- **Implementation**: `AssistantAgent` or `ConversableAgent`. Heavily reliant on executing specific tools for analysis.
- **System Prompt (Core Elements - Base Role)**:
  - "You are a meticulous Quality Control (QC) Agent."
  - "Your primary mission is to ensure all submitted work adheres strictly to the project's defined quality standards, best practices, and specific requirements for this task."
  - "Objectively analyze code, design implementations, documentation, and other artifacts using provided tools and checklists."
  - "Verify compliance with coding standards (linting, style), design system fidelity, security guidelines, accessibility requirements, and documentation completeness."
  - "Focus on identifying deviations, potential risks, and areas for improvement."
  - "Provide clear, concise, and actionable feedback reports detailing findings, referencing specific standards or requirements."
  - "Escalate critical quality issues to the QC Lead or Daedalus as defined by the process."
  - "Do not perform functional testing (that is the Tester's role); focus on static quality attributes and standard adherence."
- **Dynamic Context (Task-Specific Focus)**: Receives specific focus areas or metrics via task assignment (e.g., "Prioritize security checks," "Verify pixel-perfect implementation").
- **Composable Augments (Primarily Tool/Domain Focused Examples)**:
  - `Augment:QC_Code_StaticAnalysis`: Expertise in using advanced static analysis tools (e.g., SonarQube, Checkmarx integrations).
  - `Augment:QC_Security_Advanced`: Ability to run more in-depth security scanning tools.
  - `Augment:QC_Accessibility_Expert`: Deep knowledge and tools for WCAG compliance verification.
  - `Augment:QC_Performance_Analysis`: Tools to analyze code for potential performance bottlenecks (static analysis).
  - `Augment:QC_DesignSystem_Checker`: Tools to automatically compare implementation against design system tokens/specs.
- **Required Tools/Functions (Base Role + Augments)**:
  - `run_linter(file_paths, language)`
  - `run_static_analyzer(code_base_path, tool_config)` (Potentially via `Augment:QC_Code_StaticAnalysis`)
  - `run_security_scanner(code_base_path)` (Potentially via `Augment:QC_Security_Advanced`)
  - `compare_design_implementation(screenshot/url, design_spec_path)` (Potentially via `Augment:QC_DesignSystem_Checker`)
  - `check_accessibility(url/component)` (Potentially via `Augment:QC_Accessibility_Expert`)
  - `validate_documentation(doc_files, checklist)`
  - `generate_qc_report(findings_list, overall_status)`
  - `query_global_memory(query_text)` (Crucial for `/memory/global/best_practices`, design specs, security standards)
- **Memory Access**:
  - Read access to `/memory/global/` (best practices, design system, security standards, checklists).
  - Read access to submitted work artifacts (code, design files, docs) from `/teams/{team_id}/`.
  - Write access to a QC reporting system/memory partition.

#### 5.7. BaseDocAgent

- **Core Role**: Responsible for creating, refining, and maintaining clear, accurate, and comprehensive documentation for various audiences (users, developers, administrators). Extracts information from code, design artifacts, and agent conversations.
- **Implementation**: `AssistantAgent` or `ConversableAgent`. Needs strong natural language generation and summarization capabilities, potentially tools for interacting with code or specific documentation formats.
- **System Prompt (Core Elements - Base Role)**:
  - "You are a meticulous Technical Writer AI."
  - "Your primary goal is to produce high-quality documentation that is clear, concise, accurate, and easy for the target audience to understand."
  - "Generate various types of documentation, including user guides, API references, tutorials, setup instructions, and developer notes, based on provided source materials (code, design specs, requirements, agent conversations)."
  - "Analyze source code comments (e.g., docstrings) to generate initial API documentation."
  - "Structure documentation logically with appropriate headings, examples, and formatting."
  - "Ensure consistency in terminology and style across all documentation artifacts, adhering to project standards."
  - "Review and update existing documentation to reflect changes in the software or requirements."
- **Composable Augments (Examples)**:
  - `Augment:DocFormat_Markdown`: Expertise in GitHub Flavored Markdown and best practices.
  - `Augment:DocFormat_RestructuredText`: Expertise in reStructuredText for tools like Sphinx.
  - `Augment:DocTool_Sphinx`: Ability to configure and generate documentation using Sphinx.
  - `Augment:DocTool_SwaggerOpenAPI`: Generating API documentation in OpenAPI (Swagger) format.
  - `Augment:DocStyle_UserGuide`: Focus on writing end-user-friendly guides.
  - `Augment:DocStyle_APIRef`: Focus on detailed, accurate API reference documentation.
  - `Augment:DocSource_CodeAnalysis`: Enhanced ability to extract documentation details directly from analyzing source code.
- **Required Tools/Functions (Base Role + Augments)**:
  - `read_file(file_path)`
  - `analyze_code_comments(file_path, language)` (Potentially via `Augment:DocSource_CodeAnalysis`)
  - `generate_documentation_section(topic, source_material, format)`
  - `format_markdown(text)` (Potentially via `Augment:DocFormat_Markdown`)
  - `generate_sphinx_conf(project_details)` (Potentially via `Augment:DocTool_Sphinx`)
  - `generate_openapi_spec(api_definitions)` (Potentially via `Augment:DocTool_SwaggerOpenAPI`)
  - `query_global_memory(query_text)` (e.g., documentation standards, templates)
  - `query_team_memory(query_text)` (e.g., feature details, design specs)
  - `request_clarification(topic, agent_name)`
- **Memory Access**:
  - Read access to `/memory/global/` (documentation standards, templates).
  - Read access to `/teams/{team_id}/` (feature descriptions, code, design specs, conversation history).
  - Write access to documentation repositories or designated storage areas.

#### AgentOps Integration
- **Implementation**: Enhanced AgentOps with custom dashboard
- **Tracking Scope**:
  - Agent thoughts and reasoning
  - Inter-agent communications
  - Decision points and approvals
  - Performance metrics
  - Resource utilization

#### Custom Dashboard Elements
- **Real-time Displays**:
  1. Agent Activity Monitor
  2. Development Progress Tracker
  3. Resource Utilization Graphs
  4. Quality Metrics Dashboard
  5. Team Communication Visualizer

#### Visualization Theme
- **Design Language**: Neomorphic glass interface
- **Color Scheme**: 
  - Primary: #2A2D3E (Dark blue-grey)
  - Secondary: #00BCD4 (Cyan)
  - Accent: #FF4081 (Pink)
  - Success: #4CAF50 (Green)
  - Warning: #FFC107 (Amber)
  - Error: #F44336 (Red)

### 5. Quality Control System

#### Validation Layers
1. **Code Quality**
   - Static analysis
   - Code style enforcement
   - Performance metrics
   - Security scanning

2. **Design Excellence**
   - UI component validation
   - Accessibility compliance
   - Responsive design testing
   - Visual consistency checks

3. **Functional Testing**
   - Unit test coverage
   - Integration testing
   - End-to-end validation
   - Performance benchmarking

4. **Documentation Quality**
   - Completeness checking
   - Clarity validation
   - Example verification
   - API documentation coverage

#### Rejection Handling
- **Process Flow**:
  1. Issue Detection
  2. Automated Analysis
  3. Correction Recommendation
  4. Team Assignment
  5. Fix Implementation
  6. Re-validation

### 6. Delivery System

#### Package Components
1. **Source Code**
   - Organized repository structure
   - Build configuration files
   - Environment specifications

2. **Documentation**
   - Installation guide
   - API documentation
   - User manual
   - Development guide
   - Testing report

3. **Deployment Assets**
   - Docker containers
   - Configuration files
   - Environment variables template
   - Database schemas

4. **Quality Reports**
   - Test coverage reports
   - Performance benchmarks
   - Security audit results
   - Code quality metrics

## Discussion Points for Further Development

1. **Team Communication Pattern**
   - Group chat vs. synchronous
   - Message routing optimization
   - Inter-team communication protocols

2. **Memory Architecture**
   - Persistence strategy
   - User session management
   - Knowledge base structure

3. **Visualization Systems**
   - Development progress displays
   - Agent communication visualization
   - System status monitoring

4. **Quality Control**
   - Rejection criteria
   - Correction workflows
   - Quality metrics

## Next Steps
1. Define team structures and roles
2. Design communication patterns
3. Implement memory system
4. Develop avatar interface
5. Create monitoring dashboards

## Open Questions
1. Memory persistence strategy
2. Team communication patterns
3. Visualization implementation details
4. Quality control metrics definition

### 6. Development Roadmap (Incremental Core-First Approach)

**Note:** This section provides a high-level overview of the development phases. **The detailed, step-by-step implementation plan is maintained in the `roadmap.md` file.**

This roadmap outlines the planned development process, focusing on implementing core functionality first and incrementally adding features and complexity to ensure stability and simplify debugging.

**Phase 0: Foundation Setup**
- [ ] Set up Python project structure (e.g., using Poetry or venv).
- [ ] Initialize Git repository.
- [ ] Install core dependencies: `autogen-agentchat`, `autogen-ext[openai]`, potentially `python-dotenv`.
- [ ] Create initial `notes.md` (Done).
- [ ] Set up basic configuration management (e.g., `OAI_CONFIG_LIST.json`, `.env` file).

**Phase 1: Minimal Agent Core**
- [ ] Implement minimal `Daedalus` agent: Can receive input, print a message.
- [ ] Implement minimal `BaseDeveloperAgent`: Can receive a message, print acknowledgement.
- [ ] Establish basic communication link: Daedalus sends a message to Developer (e.g., using `initiate_chat`).
- **Goal:** Verify basic agent creation, configuration loading, and point-to-point message passing.

**Phase 2: Basic Task Execution & File I/O**
- [ ] Enhance `Daedalus`: Basic task decomposition (e.g., parse "write file X with content Y").
- [ ] Implement core file tools for `BaseDeveloperAgent`: `read_file`, `write_file`, `list_directory`.
- [ ] Enhance `Daedalus`: Assign simple file writing task to Developer.
- [ ] Enhance `BaseDeveloperAgent`: Execute the file writing task using its tools.
- **Goal:** Test basic tool implementation and execution, task assignment and completion.

**Phase 3: Group Chat Introduction**
- [ ] Implement minimal `BaseTeamLeadAgent`: Can receive/forward messages.
- [ ] Implement basic `GroupChat` configuration.
- [ ] Enhance `Daedalus`: Instantiate Lead & Dev, create `GroupChat`, assign task to Lead.
- [ ] Enhance `BaseTeamLeadAgent`: Receive task, assign/forward to Dev within `GroupChat`.
- [ ] Configure `GroupChatManager` (using the Lead agent).
- **Goal:** Test basic team formation, GroupChat mechanics, and task delegation via a Lead.

**Phase 4: Basic Memory Integration**
- [ ] Set up ChromaDB vector database (local instance initially).
- [ ] Implement `query_global_memory` tool for base agents.
- [ ] Add placeholder entries to global memory (e.g., "Coding Standard: Use snake_case").
- [ ] Enhance `BaseDeveloperAgent`: Query memory for coding standard before executing a task.
- [ ] Implement basic caching layer concept (can be simple in-memory dict initially).
- **Goal:** Test connection to memory DB, basic RAG retrieval, and context augmentation.

**Phase 5: Augmentation Concept & Prompt Engineering**
- [ ] Define 1-2 simple Augments conceptually (e.g., `Augment:Language_Python`).
- [ ] Enhance `Daedalus`: Logic to select a conceptual Augment based on task, append its (hardcoded) `PromptSnippet` to the target agent's system message during instantiation.
- **Goal:** Test the mechanism for dynamic agent specialization via prompt modification.

**Phase 6: Basic Testing & QC Roles**
- [ ] Implement minimal `BaseTesterAgent`: Can receive task (e.g., "check file X exists"), execute check, report pass/fail.
- [ ] Implement minimal `BaseQCAgent`: Can receive code path, run a basic linter tool, report findings.
- [ ] Define basic team templates (e.g., `template_medium_feature` includes Tester).
- [ ] Enhance `Daedalus`/`TeamLead`: Integrate Tester/QC steps into the workflow based on selected template.
- **Goal:** Test integration of non-developer roles and basic quality/validation steps.

**Phase 7: Basic Design & Documentation Roles**
- [ ] Implement minimal `BaseDesignerAgent`: Can receive requirements, output text description of a design.
- [ ] Implement minimal `BaseDocAgent`: Can receive code path, generate basic text description.
- [ ] Integrate Designer/Doc steps into relevant team templates/workflows.
- **Goal:** Test integration of design input and documentation output steps.

**Phase 8: Iterative Feature Expansion**
- [ ] Implement more agent tools gradually.
- [ ] Develop the structured Augment Management System (database).
- [ ] Implement robust Memory system (knowledge curation, cache invalidation).
- [ ] Implement RAG retriever agents (`RetrieveUserProxyAgent`).
- [ ] Begin development of Daedalus UI/Avatar placeholder.
- [ ] Set up AgentOps and basic Monitoring Dashboard integration.
- [ ] Refine prompts, team templates, and interaction flows based on testing.
- [ ] Implement secure code execution (Docker).

**Phase 9: Scaling & Optimization**
- [ ] Address performance bottlenecks identified during expansion.
- [ ] Test framework with larger, more complex user requests.
- [ ] Implement advanced memory features (e.g., self-improvement heuristics, potential ML model training on memory data).
- [ ] Refine Daedalus UI/Avatar and Monitoring Dashboard features.
- [ ] Conduct comprehensive end-to-end testing. 