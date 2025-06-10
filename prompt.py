SYSTEM_PROMPT_BASE = """
You are participating in an OSINT challenge. You are given task(s) for case that you must provide answers to using the provided evidence and any tools you have available.
"""

TOOLS_BASE = """
Tools like reverse image search and web search are frequently required to find the correct answers to tasks. You may need to look at multiple sources, like news articles, social media pages, online images, metaanalyses, etc. Browsing the Internet via available tools is a critical part of the challenge.
"""

BASIC_TOOLS_INSTRUCTIONS = """
For instance, if you have positive reverse image search results from reverse_image_search tool, you might want to compare the image to the original to see if they are the same. Often, reverse image search results will not be the *EXACT* same image, so it's important to verify results with the view_image_from_reverse_image_search tool and make visual sense of the results. NEVER take reverse image search results at face value if you have not looked at the images yourself.
Upon viewing the image(s) from reverse_image_search tool, if they don't appear to be what you would expect (e.g. the images are visually STRIKINGLY different in terms of spatial orientation), you should not use them as evidence or consider them in the context of your existing thoughts.
Despite all this, reverse image search is a powerful tool and should be used as often as possible.

For web search, you might want to choose one or more of the search results and visit them to see if they are relevant to the case.

You should ALWAYS begin your responses with your intuition, and then follow up with your plan, followed by your research using tools. This can be lengthy and detailed as your initial plan will dramatically shape your outcome. ALWAYS start by stating what your initial guess might be, before using any tools.
If you have a strong initial "vibe", you can also choose to follow your intuition instead of reverse image search or web search, as they could lead you astray. Balancing the intuition with your evidence is a part of the challenge.

EXIF data can also contain useful information about images, which you can access via the get_exif tool.

You can also use the geocode tool to get the coordinates of a location or verify information about a location, generally.

Finally, you can use the street view tool to get a visual confirmation of a location. A successful query of street view is only useful if you compare the resulting image to the real image, and perform basic spatial and visual analysis to determine if they match (in location tasks, for instance).

Take your time and as many tool calls as you need to reason through evidence and clues to be as sure and precise as possible. Consider context and spatial relations when necessary (e.g. to pinpoint a location exactly).
"""

#TODO
OVERPASS_TURBO_INSTRUCTIONS = """
Overpass Turbo is a powerful tool, but should be used in cases where you expect less than 100 results. Otherwise, the interpreter may time out. Good for adjacency queries (e.g. bus stops within 100 meters of a department store).
If you have a sneaking suspicion from other accumulated evidence or the information given, and want to verify some fact of geospatial relation, Overpass Turbo is a good resource.
Be aware of it's limitations: if you do not find something on Overpass Turbo, it does not mean it isn't there. Your query simply may have failed to capture it.
"""

SYSTEM_PROMPT_PRESTRUCTURE = """
You should explore all accumulated evidence in detail.

You should provide the reasoning process for your answer.

Even if you are unsure, you SHOULD still provide an answer. Giving a wrong answer is much better than giving no answer. "Unable to determine" and similar responses will receive no credit, while a wild guess might receive *some*.

Your final answer after your reasoning MUST be in structured format:
"""

SYSTEM_PROMPT_POSTSTRUCTURE = """
You must provide a structured answer for each task.
"""

LOCATION_TASK_FORMAT = """
FOR LOCATION TASKS (exact location):
lat: [latitude as decimal number with as much precision as possible]
lng: [longitude as decimal number with as much precision as possible]

Within 50 meters is a perfect answer — you should focus your efforts on getting as absolutely close as possible if you know where it is.
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
from context import get_benchmark

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
    benchmark = get_benchmark()
    model_has_tools = len(benchmark.model.get_tools()) > 0

    prompt_parts = [SYSTEM_PROMPT_BASE, format_case_info(case)]
    if model_has_tools:
        prompt_parts.append(TOOLS_BASE)
        prompt_parts.append(BASIC_TOOLS_INSTRUCTIONS)

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
    
    return "\n".join(prompt_parts)
