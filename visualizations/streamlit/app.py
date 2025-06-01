import streamlit as st
import json
import os
from PIL import Image
import pandas as pd

# Function to load metadata
def load_metadata(dataset_folder_path):
    metadata_path = os.path.join(dataset_folder_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return None

# Function to get case by ID
def get_case_by_id(metadata, case_id):
    for case in metadata.get("cases", []):
        if case.get("id") == case_id:
            return case
    return None

st.set_page_config(layout="wide")

st.title("Dataset viewer")

# Sidebar for dataset selection and case navigation
st.sidebar.header("Navigation")
dataset_base_path = "dataset"
available_datasets = [d for d in os.listdir(dataset_base_path) if os.path.isdir(os.path.join(dataset_base_path, d))]

selected_dataset_name = st.sidebar.selectbox("Select Dataset", available_datasets)

if selected_dataset_name:
    selected_dataset_path = os.path.join(dataset_base_path, selected_dataset_name)
    metadata = load_metadata(selected_dataset_path)

    if metadata:
        # Add mode selection
        view_mode = st.sidebar.radio(
            "View Mode",
            ["Browse Cases", "View All Locations"],
            index=0
        )
        
        if view_mode == "Browse Cases":
            case_ids = [case.get("id") for case in metadata.get("cases", []) if case.get("id") is not None]
            if not case_ids:
                st.sidebar.warning("No cases found in the selected dataset's metadata.json.")
                st.warning(f"No cases (or cases with IDs) found in {selected_dataset_name}/metadata.json")
            else:
                selected_case_id_str = st.sidebar.radio("Select Case ID", [str(cid) for cid in sorted(case_ids)])
                selected_case_id = int(selected_case_id_str)
                
                case_data = get_case_by_id(metadata, selected_case_id)

                if case_data:
                    # Main area to display images and JSON
                    col1, spacer, col2 = st.columns([2, 0.15, 2])

                    with col1:
                        if case_data.get("images"):
                            for img_path_suffix in case_data["images"]:
                                full_img_path = os.path.join(selected_dataset_path, img_path_suffix)
                                if os.path.exists(full_img_path):
                                    try:
                                        image = Image.open(full_img_path)
                                        st.image(image, caption=os.path.basename(full_img_path), use_column_width=True)
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

                        if case_data.get("tasks"):
                            for i, task in enumerate(case_data["tasks"]):
                                st.markdown(f"### Task {i+1}")
                                if task.get("type"):
                                    st.markdown(f"**Type:** {task['type']}")
                                if task.get("prompt"):
                                    st.markdown(f"**Prompt:** {task['prompt']}")
                                if task.get("answer") is not None:
                                    st.markdown("**Answer:**")
                                    if isinstance(task["answer"], dict):
                                        st.json(task["answer"])
                                    else:
                                        st.write(task["answer"])
                                if task.get("note"):
                                    st.markdown(f"**Note:** {task['note']}")
                                st.markdown("---")
                    
                    # Map section for location tasks
                    location_tasks = [task for task in case_data.get("tasks", []) if task.get("type") == "location" and task.get("answer")]
                    if location_tasks:
                        st.markdown("---")
                        st.header("📍 Task Locations")
                        
                        map_data = []
                        for i, task in enumerate(location_tasks):
                            answer = task.get("answer", {})
                            if isinstance(answer, dict) and "lat" in answer and "lng" in answer:
                                try:
                                    lat = float(answer["lat"])
                                    lng = float(answer["lng"])
                                    map_data.append({
                                        "lat": lat,
                                        "lon": lng,
                                        "task": f"Task {case_data['tasks'].index(task) + 1}"
                                    })
                                except (ValueError, TypeError):
                                    continue
                        
                        if map_data:
                            map_df = pd.DataFrame(map_data)
                            st.map(map_df)
                            
                            # Show coordinates as text too
                            st.markdown("**Coordinates:**")
                            for point in map_data:
                                st.write(f"• {point['task']}: {point['lat']}, {point['lon']}")
                else:
                    st.error(f"Case ID {selected_case_id} not found in metadata.")
        else:
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
    else:
        st.error(f"Could not load metadata.json from {selected_dataset_path}. Make sure the file exists and is correctly formatted.")
else:
    st.info("Select a dataset from the sidebar to begin.")