SYSTEM_PROMPT_BASE = """
You are participating in an OSINT challenge. You are given task(s) that you must provide answers to using the provided evidence and any tools you have available.
For instance, you have access to Google Search, which is OFTEN required to give answers to tasks. You may need to look at multiple sources, like news articles, social media pages, online images, etc.

Take your time to reason through evidence and clues; you should provide the reasoning for your answer.

"""

SYSTEM_PROMPT_PRESTRUCTURE = """
Your final answer after your reasoning MUST be in structured format:

"""

SYSTEM_PROMPT_POSTSTRUCTURE = """
You must provide a structured answer for each task, BUT you should only provide a structured format for the task types you are given. For instance, do not provide a temporal task answer if there is not a temporal task.
"""

LOCATION_TASK_FORMAT = """
FOR LOCATION TASKS:
lat: [latitude as decimal number with as much precision as possible]
lng: [longitude as decimal number with as much precision as possible]

Within 100 meters is a perfect answer — you should focus your efforts on getting as absolutely close as possible if you know where it is.
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
from tools import get_exif_data

def format_case_info(case) -> str:
    """Format case information as a text string."""
    case_info = case.info + "\n"

    if case.images:
        for img in case.images:
            exif = get_exif_data(img)
            if exif:
                case_info += f"Here is the EXIF data for the image: <exif>\n{exif}\n</exif>"

    return f"<info>{case_info}</info>"

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

    with open("prompt.txt", "w") as f:
        f.write(system_prompt("\n".join(prompt_parts)))
    
    return system_prompt("\n".join(prompt_parts))

def system_prompt(prompt_string: str) -> str:
    return "<system>" + prompt_string + "</system>"
