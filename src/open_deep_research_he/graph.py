from typing import Literal, Dict, Any, List, Optional, Type, TypeVar, cast, Union
import asyncio
import json
import re

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

from open_deep_research_he.state import (
    ReportStateInput,
    ReportStateOutput,
    Sections,
    ReportState,
    SectionState,
    SectionOutputState,
    Queries,
    Feedback
)

# Define a type variable for the Pydantic model
T = TypeVar('T', bound=BaseModel)

# Custom function to handle structured output when with_structured_output is not available
def custom_structured_output(model, pydantic_schema: Type[T]):
    """
    A workaround for models that don't support with_structured_output method.
    Creates a similar functionality by using a PydanticOutputParser.
    
    Args:
        model: The language model
        pydantic_schema: The Pydantic schema for structured output
        
    Returns:
        A runnable that parses output into the specified schema
    """
    parser = PydanticOutputParser(pydantic_object=pydantic_schema)
    
    # Create a template that instructs the model to output in the required format
    schema_str = str(pydantic_schema.schema())
    
    # Get field descriptions for better prompting
    fields_info = []
    for field_name, field in pydantic_schema.__fields__.items():
        description = field.field_info.description or f"The {field_name}"
        fields_info.append(f"- {field_name}: {description}")
    
    field_descriptions = "\n".join(fields_info)
    
    # Create a wrapper class with invoke and ainvoke methods
    class StructuredOutputRunnable:
        def __init__(self, model, schema):
            self.model = model
            self.schema = schema
            self.parser = PydanticOutputParser(pydantic_object=schema)
        
        def invoke(self, messages: List[BaseMessage], **kwargs) -> T:
            # Use asyncio to run the async function
            return asyncio.run(self.ainvoke(messages, **kwargs))
        
        async def ainvoke(self, messages: List[BaseMessage], **kwargs) -> T:
            # Add formatting instructions to the last message
            format_instructions = f"\n\nYou must respond with a JSON object that conforms to this schema:\n{schema_str}\n\nField descriptions:\n{field_descriptions}\n\nYour response must be valid JSON without any explanations or conversation outside the JSON."
            
            # Create a copy of messages to avoid modifying the original
            messages_copy = messages.copy()
            
            # Add the formatting instructions to the last message
            if isinstance(messages_copy[-1], HumanMessage):
                new_content = messages_copy[-1].content + format_instructions
                messages_copy[-1] = HumanMessage(content=new_content)
            else:
                messages_copy.append(HumanMessage(content=format_instructions))
            
            # Call the model
            response = await self.model.ainvoke(messages_copy, **kwargs)
            
            # Parse the response
            try:
                # Extract JSON from the response
                content = response.content
                
                # Try to find JSON in the response
                json_match = re.search(r'```json\n(.+?)\n```|\{.+\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
                    # Clean up the JSON string
                    json_str = json_str.strip()
                    # Parse as JSON first to validate
                    parsed_json = json.loads(json_str)
                    # Then use the parser to convert to the Pydantic model
                    return cast(T, self.schema.parse_obj(parsed_json))
                else:
                    # If no JSON pattern found, try direct parsing
                    return cast(T, self.schema.parse_obj(json.loads(content)))
            except Exception as e:
                # If parsing fails, try a more aggressive approach
                try:
                    # Look for anything that might be JSON
                    potential_json = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                    if potential_json:
                        json_str = potential_json.group(0)
                        parsed_json = json.loads(json_str)
                        return cast(T, self.schema.parse_obj(parsed_json))
                    else:
                        raise ValueError(f"Could not extract JSON from response: {content}")
                except Exception as inner_e:
                    raise ValueError(f"Failed to parse structured output: {str(e)}\nResponse content: {content}")
        
        # Support method chaining like with_retry
        def with_retry(self, **kwargs):
            # Return self for simple implementation
            # In a real implementation, you would create a proper retry wrapper
            return self
    
    return StructuredOutputRunnable(model, pydantic_schema)

from open_deep_research_he.prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions, 
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs
)

from open_deep_research_he.configuration import WorkflowConfiguration
from open_deep_research_he.utils import (
    format_sections,
    format_sections_dict,
    get_config_value, 
    get_search_params, 
    select_and_execute_search,
    get_today_str,
    async_init_chat_model
)

from open_deep_research_he.structured_output_workaround import with_structured_output_safe

## Nodes -- 

async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """Generate the initial report plan with sections.
    
    This node:
    1. Gets configuration for the report structure and search parameters
    2. Generates search queries to gather context for planning
    3. Performs web searches using those queries
    4. Uses an LLM to generate a structured plan with sections
    
    Args:
        state: Current graph state containing the report topic
        config: Configuration for models, search APIs, etc.
        
    Returns:
        Dict containing the generated sections
    """

    # Inputs
    topic = state["topic"]

    # Get list of feedback on the report plan
    feedback_list = state.get("feedback_on_report_plan", [])

    # Concatenate feedback on the report plan into a single string
    feedback = " /// ".join(feedback_list) if feedback_list else ""

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Convert JSON object to string if necessary
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # Set writer model (model used for query writing)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    # Use async_init_chat_model to prevent blocking the event loop
    writer_model = await async_init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs)
    structured_llm = with_structured_output_safe(writer_model, Queries)

    # Format system instructions
    system_instructions_query = report_planner_query_writer_instructions.format(
        topic=topic,
        report_organization=report_structure,
        number_of_queries=number_of_queries,
        today=get_today_str()
    )

    # Generate queries  
    results = await structured_llm.ainvoke([SystemMessage(content=system_instructions_query),
                                     HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])

    # Web search
    query_list = [query.search_query for query in results.queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    # Format system instructions
    system_instructions_sections = report_planner_instructions.format(topic=topic, report_organization=report_structure, context=source_str, feedback=feedback)

    # Set the planner
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    planner_model_kwargs = get_config_value(configurable.planner_model_kwargs or {})

    # Report planner instructions
    planner_message = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
                        Each section must have: name, description, research, and content fields."""

    # Run the planner
    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        # Use async_init_chat_model to prevent blocking the event loop
        planner_llm = await async_init_chat_model(model=planner_model, 
                                      model_provider=planner_provider, 
                                      max_tokens=100_000, 
                                      thinking={"type": "enabled", "budget_tokens": 80_000})

    else:
        # With other models, thinking tokens are not specifically allocated
        # Use async_init_chat_model to prevent blocking the event loop
        planner_llm = await async_init_chat_model(model=planner_model, 
                                      model_provider=planner_provider,
                                      model_kwargs=planner_model_kwargs)
    
    # Generate the report sections
    structured_llm = with_structured_output_safe(planner_llm, Sections)
    report_sections = await structured_llm.ainvoke([SystemMessage(content=system_instructions_sections),
                                             HumanMessage(content=planner_message)])

    # Get sections
    sections = report_sections.sections

    return {"sections": sections}

def human_feedback(state: ReportState, config: RunnableConfig) -> Command[Literal["generate_report_plan","build_section_with_web_research"]]:
    """Get human feedback on the report plan and route to next steps.
    
    This node:
    1. Formats the current report plan for human review
    2. Gets feedback via an interrupt
    3. Routes to either:
       - Section writing if plan is approved
       - Plan regeneration if feedback is provided
    
    Args:
        state: Current graph state with sections to review
        config: Configuration for the workflow
        
    Returns:
        Command to either regenerate plan or start section writing
    """
    # Extract sections and topic from state
    sections = state["sections"]
    topic = state["topic"]
    
    # Debug: Print state and sections
    print(f"\n\n==== DEBUG: human_feedback state ====")
    print(f"State keys: {state.keys()}")
    print(f"Topic: {topic}")
    print(f"Number of sections: {len(sections)}")
    print(f"Sections: {sections}")
    for i, s in enumerate(sections):
        print(f"Section {i}: {s.name}, research={s.research}, type={type(s.research)}")
    
    # Format sections for display
    sections_str = "\n\n".join([f"Section: {s.name}\nDescription: {s.description}\nRequires Research: {s.research}" for s in sections])
    
    # Create interrupt message
    interrupt_message = f"""Here's the report plan for '{topic}':
                        \n\n{sections_str}\n
                        \nDoes the report plan meet your needs?\nPass 'true' to approve the report plan.\nOr, provide feedback to regenerate the report plan:"""
    
    feedback = interrupt(interrupt_message)
    print(f"\n==== DEBUG: Received feedback: {feedback} (type: {type(feedback)}) ====")

    # Extract approval from feedback
    is_approved = False
    
    # Handle dictionary feedback with UUID keys (from GUI)
    if isinstance(feedback, dict):
        print(f"\n==== DEBUG: Processing dictionary feedback: {feedback} ====")
        # Extract the value from the dictionary (regardless of the key)
        if feedback:
            # Get the first value from the dictionary
            first_value = next(iter(feedback.values()))
            print(f"Extracted value from dict: {first_value} (type: {type(first_value)})")
            
            # Check if the value indicates approval
            if isinstance(first_value, str) and first_value.lower().strip() in ['true', 'yes', 'approve', 'approved']:
                is_approved = True
                print(f"Dictionary feedback approved: {first_value}")
            elif isinstance(first_value, bool) and first_value is True:
                is_approved = True
                print(f"Dictionary feedback approved: {first_value}")
    
    # Handle direct boolean or string feedback
    elif (isinstance(feedback, bool) and feedback is True) or \
         (isinstance(feedback, str) and feedback.lower().strip() in ['true', 'yes', 'approve', 'approved']):
        is_approved = True
        print(f"Direct feedback approved: {feedback}")
    
    # If approved, kick off section writing
    if is_approved:
        print(f"\n==== DEBUG: Approval received ====")
        
        # Make sure we have sections with research=True
        research_sections = [s for s in sections if s.research]
        print(f"Research sections: {len(research_sections)} found")
        for i, s in enumerate(research_sections):
            print(f"Research section {i}: {s.name}")
            
        if not research_sections:
            print("WARNING: No research sections found. Adding research=True to all sections.")
            # If no sections have research=True, treat all sections as research sections
            research_sections = sections
            
        # Create the Command with Send objects for each research section
        sends = [
            Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0}) 
            for s in research_sections
        ]
        print(f"Created {len(sends)} Send commands for research sections")
        
        # Debug the command we're about to return
        command = Command(goto=sends)
        print(f"Returning Command with goto={len(sends)} Send objects")
        return command
    
    # If the user provides feedback as a string, regenerate the report plan 
    elif isinstance(feedback, str):
        # Treat this as feedback and append it to the existing list
        return Command(goto="generate_report_plan", 
                       update={"feedback_on_report_plan": [feedback]})
    
    # Handle dictionary feedback (common when using structured output)
    elif isinstance(feedback, dict):
        # Extract feedback from the dictionary if possible, or use the whole dict as feedback
        feedback_str = feedback.get('feedback', str(feedback))
        return Command(goto="generate_report_plan", 
                       update={"feedback_on_report_plan": [feedback_str]})
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")
    
async def generate_queries(state: SectionState, config: RunnableConfig):
    """Generate search queries for researching a specific section.
    
    This node uses an LLM to generate targeted search queries based on the 
    section topic and description.
    
    Args:
        state: Current state containing section details
        config: Configuration including number of queries to generate
        
    Returns:
        Dict containing the generated search queries
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries 
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    # Use async_init_chat_model to prevent blocking the event loop
    writer_model = await async_init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs)
    structured_llm = with_structured_output_safe(writer_model, Queries)

    # Format system instructions
    system_instructions = query_writer_instructions.format(topic=topic, 
                                                           section_topic=section.description, 
                                                           number_of_queries=number_of_queries,
                                                           today=get_today_str())

    # Generate queries  
    queries = await structured_llm.ainvoke([SystemMessage(content=system_instructions),
                                     HumanMessage(content="Generate search queries on the provided topic.")])

    return {"search_queries": queries.queries}

async def search_web(state: SectionState, config: RunnableConfig):
    """Execute web searches for the section queries.
    
    This node:
    1. Takes the generated queries
    2. Executes searches using configured search API
    3. Formats results into usable context
    
    Args:
        state: Current state with search queries
        config: Search API configuration
        
    Returns:
        Dict with search results and updated iteration count
    """

    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Web search
    query_list = [query.search_query for query in search_queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}

async def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """Write a section of the report and evaluate if more research is needed.
    
    This node:
    1. Writes section content using search results
    2. Evaluates the quality of the section
    3. Either:
       - Completes the section if quality passes
       - Triggers more research if quality fails
    
    Args:
        state: Current state with search results and section info
        config: Configuration for writing and evaluation
        
    Returns:
        Command to either complete section or do more research
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)

    # Format system instructions
    section_writer_inputs_formatted = section_writer_inputs.format(topic=topic, 
                                                             section_name=section.name, 
                                                             section_topic=section.description, 
                                                             context=source_str, 
                                                             section_content=section.content)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = await async_init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 

    section_content = await writer_model.ainvoke([SystemMessage(content=section_writer_instructions),
                                           HumanMessage(content=section_writer_inputs_formatted)])
    
    # Write content to the section object  
    section.content = section_content.content

    # Grade prompt 
    section_grader_message = ("Grade the report and consider follow-up questions for missing information. "
                              "If the grade is 'pass', return empty strings for all follow-up queries. "
                              "If the grade is 'fail', provide specific search queries to gather missing information.")
    
    section_grader_instructions_formatted = section_grader_instructions.format(topic=topic, 
                                                                               section_topic=section.description,
                                                                               section=section.content, 
                                                                               number_of_follow_up_queries=configurable.number_of_queries)

    # Use planner model for reflection
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    planner_model_kwargs = get_config_value(configurable.planner_model_kwargs or {})

    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        # First await the coroutine to get the model object
        model = await async_init_chat_model(model=planner_model, 
                                            model_provider=planner_provider, 
                                            max_tokens=100_000, 
                                            thinking={"type": "enabled", "budget_tokens": 80_000})
        # Then call with_structured_output_safe on the model object
        reflection_model = with_structured_output_safe(model, Feedback)
    else:
        # First await the coroutine to get the model object
        model = await async_init_chat_model(model=planner_model, 
                                           model_provider=planner_provider, model_kwargs=planner_model_kwargs)
        # Then call with_structured_output_safe on the model object
        reflection_model = with_structured_output_safe(model, Feedback)
    # Generate feedback
    feedback = await reflection_model.ainvoke([SystemMessage(content=section_grader_instructions_formatted),
                                        HumanMessage(content=section_grader_message)])

    # If the section is passing or the max search depth is reached, publish the section to completed sections 
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        # Publish the section to completed sections 
        update = {"completed_sections": [section]}
        if configurable.include_source_str:
            update["source_str"] = source_str
        return Command(update=update, goto=END)

    # Update the existing section with new content and update search queries
    else:
        return Command(
            update={"search_queries": feedback.follow_up_queries, "section": section},
            goto="search_web"
        )
    
async def write_final_sections(state: SectionState, config: RunnableConfig):
    """Write sections that don't require research using completed sections as context.
    
    This node handles sections like conclusions or summaries that build on
    the researched sections rather than requiring direct research.
    
    Args:
        state: Current state with completed sections as context
        config: Configuration for the writing model
        
    Returns:
        Dict containing the newly written section
    """

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)

    # Get state 
    topic = state["topic"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    
    # Format system instructions
    system_instructions = final_section_writer_instructions.format(topic=topic, section_name=section.name, section_topic=section.description, context=completed_report_sections)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = await async_init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    
    section_content = await writer_model.ainvoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="Generate a report section based on the provided sources.")])
    
    # Write content to section 
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}

def gather_completed_sections(state: ReportState):
    """Format completed sections as context for writing final sections.
    
    This node takes all completed research sections and formats them into
    a single context string for writing summary sections.
    
    Args:
        state: Current state with completed sections
        
    Returns:
        Dict with formatted sections as context
    """

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    # Using format_sections for Section objects
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}

def compile_final_report(state: ReportState, config: RunnableConfig):
    """Compile all sections into the final report.
    
    This node:
    1. Gets all completed sections
    2. Orders them according to original plan
    3. Combines them into the final report with proper structure
    4. Ensures the report reflects the detailed plan and incorporates feedback
    
    Args:
        state: Current state with all completed sections
        
    Returns:
        Dict containing the complete report
    """

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)

    # Get sections and topic
    topic = state["topic"]
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    # Get feedback that was incorporated into the plan
    feedback_list = state.get("feedback_on_report_plan", [])
    feedback_str = "\n\n".join([f"- {feedback}" for feedback in feedback_list]) if feedback_list else ""

    # Create report header with topic and metadata
    report_header = f"# {topic}\n\n"
    
    # Add feedback information if available
    if feedback_str:
        report_header += f"**Report Structure Feedback Incorporated:**\n{feedback_str}\n\n"
    
    # Add table of contents
    toc = "## Table of Contents\n\n"
    for i, section in enumerate(sections):
        toc += f"{i+1}. [{section.name}](#{section.name.lower().replace(' ', '-')})\n"
    toc += "\n\n"

    # Update sections with completed content while maintaining original order
    formatted_sections = []
    for section in sections:
        if section.name in completed_sections:
            # Get the content, ensuring it starts with the proper section header
            content = completed_sections[section.name]
            
            # Only add the section header if it's not already there
            if not content.strip().startswith(f"## {section.name}"):
                content = f"## {section.name}\n\n{content}"
            
            # Add the section description as a subtitle if not already included
            if section.description and section.description not in content:
                # Insert description after the header but before the content
                header_end = content.find("\n", content.find("##"))
                if header_end != -1:
                    content = f"{content[:header_end+1]}*{section.description}*\n\n{content[header_end+1:]}"                
            
            formatted_sections.append(content)

    # Compile final report with header, TOC, and sections
    all_sections = "\n\n".join(formatted_sections)
    final_report = f"{report_header}{toc}{all_sections}"

    if configurable.include_source_str:
        return {"final_report": final_report, "source_str": state["source_str"]}
    else:
        return {"final_report": final_report}

def initiate_final_section_writing(state: ReportState):
    """Create parallel tasks for writing non-research sections.
    
    This edge function identifies sections that don't need research and
    creates parallel writing tasks for each one.
    
    Args:
        state: Current state with all sections and research context
        
    Returns:
        List of Send commands for parallel section writing
    """

    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send("write_final_sections", {"topic": state["topic"], "section": s, "report_sections_from_research": state["report_sections_from_research"]}) 
        for s in state["sections"] 
        if not s.research
    ]

# Report section sub-graph -- 

# Add nodes 
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# Outer graph for initial report plan compiling results from each section -- 

# Add nodes
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=WorkflowConfiguration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()
