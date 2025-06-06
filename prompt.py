SYSTEM_PROMPT_BASE = """
You are participating in an OSINT challenge. You are given task(s) that you must provide answers to using the provided evidence and any tools you have available.
For instance, you have access to Search, which is OFTEN required to give answers to tasks. You may need to look at multiple sources, like news articles, social media pages, online images, etc.
You should explore the evidence in detail. For instance, if you have positive reverse image search results, you might want to compare the image to the original to see if they are the same.
EXIF data can also contain useful information about images, which is provided to you if available. However, it may not always be available.

Take your time to reason through evidence and clues; you should provide the reasoning for your answer.

Even if you are unsure, you SHOULD still provide an answer. Giving a wrong answer is much better than giving no answer. "Unable to determine" will receive no credit, while a wild guess might receive *some*.
"""

SYSTEM_PROMPT_PRESTRUCTURE = """
Your final answer after your reasoning MUST be in structured format:
"""

SYSTEM_PROMPT_POSTSTRUCTURE = """
You must provide a structured answer for each task.
"""

LOCATION_TASK_FORMAT = """
FOR LOCATION TASKS (exact location):
lat: [latitude as decimal number with as much precision as possible]
lng: [longitude as decimal number with as much precision as possible]

Within 100 meters is a perfect answer — you should focus your efforts on getting as absolutely close as possible if you know where it is.
"""

LOCATION_TASK_BETA = """
FOR LOCATION TASKS (island, city, region, etc.):
name: [name of the location]
"""

IDENTIFICATION_TASK_FORMAT = """
FOR IDENTIFICATION TASKS:
type: [entity type - person/organization/vehicle/etc]
name: [specific name if identifiable]
"""

TEMPORAL_TASK_FORMAT = """
FOR TEMPORAL TASKS:
date: [YYYY-MM-DD or descriptive period]
time: [HH:MM or time period if applicable]
"""

ANALYSIS_TASK_FORMAT = """
FOR ANALYSIS TASKS:
conclusion: [conclusion to a question - must be ONE answer, no hedging (you cannot say 'or')]
"""

import os

def format_case_info(case) -> str:
    """Format case information as a text string."""
    case_info = case.info + "\n"

    if case.images:
        for img in case.images:
            img_name = os.path.basename(img)
            case_info += f"Image available: {img_name}\n"

    return f"<info>{case_info}</info>\n"

def format_case_tasks(case) -> list[str]:
    """Format case tasks as a list of text strings."""
    return [f"<task>{task.type}: {task.prompt}</task>" for task in case.tasks]

def get_prompt(case) -> str:
    """Builds prompt for a case."""
    prompt_parts = [SYSTEM_PROMPT_BASE, format_case_info(case)]
    prompt_parts.extend(format_case_tasks(case))
    prompt_parts.append(SYSTEM_PROMPT_PRESTRUCTURE)
    
    task_types_in_case = {task.type for task in case.tasks}
    
    if "location" in task_types_in_case:
        prompt_parts.append(LOCATION_TASK_FORMAT)
    if "identification" in task_types_in_case:
        prompt_parts.append(IDENTIFICATION_TASK_FORMAT)
    if "temporal" in task_types_in_case:
        prompt_parts.append(TEMPORAL_TASK_FORMAT)
    if "analysis" in task_types_in_case:
        prompt_parts.append(ANALYSIS_TASK_FORMAT)
    
    prompt_parts.append(SYSTEM_PROMPT_POSTSTRUCTURE)

    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(system_prompt("\n".join(prompt_parts)))
    
    return system_prompt("\n".join(prompt_parts))

def system_prompt(prompt_string: str) -> str:
    return "<system>" + prompt_string + "</system>"
