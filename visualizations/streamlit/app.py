import streamlit as st
import json
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import math

# Function to load metadata
def load_metadata(dataset_folder_path):
    metadata_path = os.path.join(dataset_folder_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# Function to get case by ID
def get_case_by_id(metadata, case_id):
    for case in metadata.get("cases", []):
        if case.get("id") == case_id:
            return case
    return None

# Function to get available response runs for a dataset
def get_response_runs(dataset_name):
    responses_dir = "responses"
    if not os.path.exists(responses_dir):
        return []
    
    runs = []
    for folder in os.listdir(responses_dir):
        if folder.startswith('.'):  # Skip hidden folders
            continue
        folder_path = os.path.join(responses_dir, folder)
        if os.path.isdir(folder_path):
            # Parse folder name: {model_name}_{dataset}_{timestamp}
            parts = folder.split('_')
            if len(parts) >= 3:
                # Check if this run is for the current dataset
                # Look for dataset name in the folder name
                if dataset_name in folder:
                    # Create compact display name
                    # Extract model name (everything before dataset name)
                    model_part = folder.split(f'_{dataset_name}_')[0]
                    # Extract timestamp (everything after dataset name)
                    timestamp_part = folder.split(f'_{dataset_name}_')[1] if f'_{dataset_name}_' in folder else parts[-1]
                    
                    # Abbreviate common model names
                    model_compact = model_part.replace('Gemini 2.5 Flash Preview', 'Gemini 2.5 Flash') \
                                              .replace('Gemini 2.5 Pro', 'Gemini 2.5 Pro') \
                                              .replace('GPT-4', 'GPT-4') \
                                              .replace('Claude', 'Claude')
                    
                    # Format timestamp more compactly (from 2025-06-01T12_50_38 to 06/01 12:50)
                    try:
                        if 'T' in timestamp_part:
                            date_part, time_part = timestamp_part.split('T')
                            year, month, day = date_part.split('-')
                            hour, minute = time_part.replace('_', ':').split(':')[:2]
                            compact_time = f"{month}/{day} {hour}:{minute}"
                        else:
                            compact_time = timestamp_part[:10]  # fallback
                    except:
                        compact_time = timestamp_part[:10]  # fallback to first 10 chars
                    
                    display_name = f"{model_compact} • {compact_time}"
                    
                    runs.append({
                        'folder': folder,
                        'display_name': display_name,
                        'path': folder_path,
                        'original_name': folder.replace('_', ' ')  # Keep original for tooltips if needed
                    })
    
    return sorted(runs, key=lambda x: x['folder'], reverse=True)

# Function to get available case IDs for a specific model run
def get_available_case_ids_for_run(run_path):
    """Get list of case IDs that have output files in the given run"""
    if not run_path:
        return []
    
    output_dir = os.path.join(run_path, "output")
    if not os.path.exists(output_dir):
        return []
    
    case_ids = []
    for filename in os.listdir(output_dir):
        if filename.endswith('.txt'):
            try:
                # Extract case ID from filename (e.g., "11.txt" -> 11)
                case_id = int(os.path.splitext(filename)[0])
                case_ids.append(case_id)
            except ValueError:
                continue  # Skip files that don't have numeric names
    
    return sorted(case_ids)

# Function to load model response for a case
def load_model_response(run_path, case_id):
    output_file = os.path.join(run_path, "output", f"{case_id}.txt")
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error loading response: {e}"
    return None

# Function to parse location from model response
def parse_location_from_response(response_text):
    """Try to extract lat/lng coordinates from model response"""
    if not response_text:
        return None
    
    lines = response_text.strip().split('\n')
    lat, lng = None, None
    
    # Look for lat/lng patterns - check lines in reverse order as coordinates are often at the end
    for line in reversed(lines):
        line = line.strip()
        
        # Handle various coordinate formats
        if line.startswith('lat:') or line.startswith('latitude:'):
            try:
                lat = float(line.split(':')[1].strip())
            except (ValueError, IndexError):
                continue
        elif line.startswith('lng:') or line.startswith('longitude:') or line.startswith('lon:'):
            try:
                lng = float(line.split(':')[1].strip())
            except (ValueError, IndexError):
                continue
        
        # Also check for patterns like "lat: 39.6265" without additional text
        if line.lower().startswith('lat') and ':' in line:
            try:
                lat_str = line.split(':')[1].strip()
                lat = float(lat_str)
            except (ValueError, IndexError):
                continue
        elif line.lower().startswith('lng') or line.lower().startswith('lon'):
            if ':' in line:
                try:
                    lng_str = line.split(':')[1].strip()
                    lng = float(lng_str)
                except (ValueError, IndexError):
                    continue
    
    if lat is not None and lng is not None:
        return {"lat": lat, "lng": lng}
    
    return None

st.set_page_config(layout="wide")

# Custom CSS to make sidebar wider and disable map interaction
st.markdown("""
<style>
    .css-1d391kg {
        width: 400px;
    }
    .css-1lcbmhc {
        width: 400px;
    }
    .css-17eq0hr {
        width: 400px;
    }
    section[data-testid="stSidebar"] {
        width: 400px !important;
    }
    section[data-testid="stSidebar"] > div {
        width: 400px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Dataset viewer")

# Sidebar for dataset selection and case navigation
st.sidebar.header("Navigation")
dataset_base_path = "dataset"
available_datasets = [d for d in os.listdir(dataset_base_path) if os.path.isdir(os.path.join(dataset_base_path, d)) and not d.startswith('.')]

selected_dataset_name = st.sidebar.selectbox("Select Dataset", available_datasets)

if selected_dataset_name:
    selected_dataset_path = os.path.join(dataset_base_path, selected_dataset_name)
    metadata = load_metadata(selected_dataset_path)

    if metadata:
        # Get available response runs for this dataset
        response_runs = get_response_runs(selected_dataset_name)
        
        # Response dropdown (only show if there are runs available)
        selected_run = None
        if response_runs:
            st.sidebar.header("Model Responses")
            run_options = ["None (Ground Truth Only)"] + [run['display_name'] for run in response_runs]
            selected_run_name = st.sidebar.selectbox("Select Model Run", run_options)
            
            if selected_run_name != "None (Ground Truth Only)":
                selected_run = next(run for run in response_runs if run['display_name'] == selected_run_name)
        
        # Add mode selection
        view_mode = st.sidebar.radio(
            "View Mode",
            ["Browse Cases", "View All Locations", "Task Distribution"],
            index=0
        )
        
        if view_mode == "Browse Cases":
            # Get all case IDs from metadata
            all_case_ids = [case.get("id") for case in metadata.get("cases", []) if case.get("id") is not None]
            
            # Filter case IDs based on selected model run
            if selected_run:
                # Only show cases that this model run actually processed
                available_case_ids = get_available_case_ids_for_run(selected_run['path'])
                # Filter to only include cases that exist in both metadata and model output
                case_ids = [cid for cid in available_case_ids if cid in all_case_ids]
                
                if not case_ids:
                    st.sidebar.warning(f"No cases found for the selected model run.")
                    st.warning(f"The selected model run '{selected_run['display_name']}' has no processed cases.")
            else:
                # No model selected, show all cases from metadata
                case_ids = all_case_ids
            
            if not case_ids:
                if not selected_run:
                    st.sidebar.warning("No cases found in the selected dataset's metadata.json.")
                    st.warning(f"No cases (or cases with IDs) found in {selected_dataset_name}/metadata.json")
            else:
                selected_case_id_str = st.sidebar.radio("Select Case ID", [str(cid) for cid in sorted(case_ids)])
                selected_case_id = int(selected_case_id_str)
                
                case_data = get_case_by_id(metadata, selected_case_id)

                if case_data:
                    # Load model response if a run is selected
                    model_response = None
                    if selected_run:
                        model_response = load_model_response(selected_run['path'], selected_case_id)
                    
                    # Main area to display images and JSON
                    col1, spacer, col2 = st.columns([2, 0.15, 2])

                    with col1:
                        if case_data.get("images"):
                            for img_path_suffix in case_data["images"]:
                                full_img_path = os.path.join(selected_dataset_path, img_path_suffix)
                                if os.path.exists(full_img_path):
                                    try:
                                        image = Image.open(full_img_path)
                                        st.image(image, caption=os.path.basename(full_img_path), use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error loading image {os.path.basename(full_img_path)}: {e}")
                                else:
                                    st.warning(f"Image not found: {full_img_path}")
                        else:
                            st.write("No images specified for this case.")

                    with col2:
                        st.header(f"Details for Case {selected_case_id}")
                        
                        if case_data.get("info"):
                            st.write(case_data["info"])
                            st.markdown("---")

                        # Show model response if available
                        if model_response:
                            st.subheader("🤖 Model Response")
                            with st.expander(f"Response from {selected_run['display_name']}", expanded=True):
                                st.text_area("Raw Response", model_response, height=200)
                            st.markdown("---")

                        if case_data.get("tasks"):
                            for i, task in enumerate(case_data["tasks"]):
                                st.markdown(f"### Task {i+1}")
                                if task.get("type"):
                                    st.markdown(f"**Type:** {task['type']}")
                                if task.get("prompt"):
                                    st.markdown(f"**Prompt:** {task['prompt']}")
                                if task.get("answer") is not None:
                                    st.markdown("**Ground Truth Answer:**")
                                    if isinstance(task["answer"], dict):
                                        st.json(task["answer"])
                                    else:
                                        st.write(task["answer"])
                                if task.get("note"):
                                    st.markdown(f"**Note:** {task['note']}")
                                st.markdown("---")
                    
                    # Map section for location tasks
                    location_tasks = [task for task in case_data.get("tasks", []) if task.get("type") == "location" and task.get("answer")]
                    model_location = None
                    if model_response:  # Check for model location regardless of whether there are ground truth location tasks
                        model_location = parse_location_from_response(model_response)
                    
                    if location_tasks or model_location:
                        st.markdown("---")
                        st.header("Task Locations")
                        
                        map_data = []
                        
                        # Add ground truth locations
                        for i, task in enumerate(location_tasks):
                            answer = task.get("answer", {})
                            if isinstance(answer, dict) and "lat" in answer and "lng" in answer:
                                try:
                                    lat = float(answer["lat"])
                                    lng = float(answer["lng"])
                                    map_data.append({
                                        "lat": lat,
                                        "lon": lng,
                                        "task": f"Ground Truth - Task {case_data['tasks'].index(task) + 1}",
                                        "color": [255, 0, 0, 200]  # Red with transparency
                                    })
                                except (ValueError, TypeError):
                                    continue
                        
                        # Add model prediction location if available
                        if model_location and selected_run:
                            map_data.append({
                                "lat": model_location["lat"],
                                "lon": model_location["lng"],
                                "task": f"Model Prediction - {selected_run['display_name']}",
                                "color": [0, 100, 255, 200]  # Blue with transparency
                            })
                        
                        if map_data:
                            map_df = pd.DataFrame(map_data)
                            st.map(map_df, size=10000, color="color", zoom=6)
                        
                        # Show distance if both ground truth and model prediction exist
                        if len(map_data) >= 2 and any("Ground Truth" in p['task'] for p in map_data) and any("Model Prediction" in p['task'] for p in map_data):
                            ground_truth = next(p for p in map_data if "Ground Truth" in p['task'])
                            model_pred = next(p for p in map_data if "Model Prediction" in p['task'])
                            
                            # Calculate distance using Haversine formula (simple approximation)
                            def haversine_distance(lat1, lon1, lat2, lon2):
                                # Convert latitude and longitude from degrees to radians
                                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                                
                                # Haversine formula
                                dlat = lat2 - lat1
                                dlon = lon2 - lon1
                                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                                c = 2 * math.asin(math.sqrt(a))
                                r = 6371  # Radius of earth in kilometers
                                return c * r
                            
                            distance = haversine_distance(
                                ground_truth['lat'], ground_truth['lon'],
                                model_pred['lat'], model_pred['lon']
                            )
                            
                            if distance < 1:
                                st.success(f"📏 **Distance:** {distance:.2f} km")
                            elif distance < 10:
                                st.warning(f"📏 **Distance:** {distance:.2f} km")
                            else:
                                st.error(f"📏 **Distance:** {distance:.2f} km")
                else:
                    st.error(f"Case ID {selected_case_id} not found in metadata.")
        elif view_mode == "View All Locations":
            # Implement "View All Locations" functionality
            st.header("🗺️ All Location Tasks")
            
            all_locations = []
            for case in metadata.get("cases", []):
                case_id = case.get("id")
                for task_idx, task in enumerate(case.get("tasks", [])):
                    if task.get("type") == "location" and task.get("answer"):
                        answer = task.get("answer", {})
                        if isinstance(answer, dict) and "lat" in answer and "lng" in answer:
                            try:
                                lat = float(answer["lat"])
                                lng = float(answer["lng"])
                                all_locations.append({
                                    "lat": lat,
                                    "lon": lng,
                                    "case_id": case_id,
                                    "task_num": task_idx + 1,
                                    "prompt": task.get("prompt", ""),
                                    "info": case.get("info", "")
                                })
                            except (ValueError, TypeError):
                                continue
            
            if all_locations:
                # Create DataFrame for the map
                map_df = pd.DataFrame(all_locations)
                st.map(map_df)
                
                st.markdown("---")
                st.subheader("Location Details")
                
                # Display location details in a table
                for i, loc in enumerate(all_locations):
                    with st.expander(f"Case {loc['case_id']}, Task {loc['task_num']} - {loc['lat']:.6f}, {loc['lon']:.6f}"):
                        st.write(f"**Task Prompt:** {loc['prompt']}")
                        st.write(f"**Case Info:** {loc['info']}")
                        st.write(f"**Coordinates:** {loc['lat']}, {loc['lon']}")
            else:
                st.info(f"No location tasks found in the {selected_dataset_name} dataset.")
        else:  # Task Distribution
            st.header("📊 Task Type Distribution")
            
            # Analyze all tasks across all cases
            task_counts = {}
            total_tasks = 0
            total_cases = len(metadata.get("cases", []))
            
            for case in metadata.get("cases", []):
                case_tasks = case.get("tasks", [])
                total_tasks += len(case_tasks)
                
                for task in case_tasks:
                    task_type = task.get("type", "unknown")
                    task_counts[task_type] = task_counts.get(task_type, 0) + 1
            
            # Display summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cases", total_cases)
            with col2:
                st.metric("Total Tasks", total_tasks)
            with col3:
                st.metric("Task Types", len(task_counts))
            
            if task_counts:
                st.markdown("---")
                
                # Display pie chart
                st.subheader("Task Type Distribution")
                
                # Create DataFrame explicitly
                df = pd.DataFrame(list(task_counts.items()), columns=['Task Type', 'Count'])
                
                # Center the chart in a smaller column
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.pie(df['Count'], labels=df['Task Type'], autopct='%1.1f%%')
                    ax.axis('equal')
                    st.pyplot(fig)
                
                # Show percentages
                st.subheader("Breakdown")
                for task_type, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_tasks) * 100
                    st.write(f"**{task_type}**: {count} tasks ({percentage:.1f}%)")
            else:
                st.info(f"No tasks found in the {selected_dataset_name} dataset.")
    else:
        st.error(f"Could not load metadata.json from {selected_dataset_path}. Make sure the file exists and is correctly formatted.")
else:
    st.info("Select a dataset from the sidebar to begin.")