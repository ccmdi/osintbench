import streamlit as st
import json
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import math
import html as htmlmod
import folium
from streamlit_folium import st_folium


@st.cache_data
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
@st.cache_data
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
            parts = folder.split('_')
            if len(parts) >= 3:
                if dataset_name in folder:
                    model_part = folder.split(f'_{dataset_name}_')[0]
                    timestamp_part = folder.split(f'_{dataset_name}_')[1] if f'_{dataset_name}_' in folder else parts[-1]
                    
                    model_compact = model_part.replace('Gemini 2.5 Flash Preview', 'Gemini 2.5 Flash') \
                                            .replace('Gemini 2.5 Pro', 'Gemini 2.5 Pro') \
                                            .replace('GPT-4', 'GPT-4') \
                                            .replace('Claude', 'Claude')
                    
                    try:
                        if 'T' in timestamp_part:
                            date_part, time_part = timestamp_part.split('T')
                            year, month, day = date_part.split('-')
                            hour, minute = time_part.replace('_', ':').split(':')[:2]
                            compact_time = f"{month}/{day} {hour}:{minute}"
                        else:
                            compact_time = timestamp_part[:10]
                    except:
                        compact_time = timestamp_part[:10]
                    
                    display_name = f"{model_compact} • {compact_time}"
                    
                    runs.append({
                        'folder': folder,
                        'display_name': display_name,
                        'path': folder_path,
                        'original_name': folder.replace('_', ' ')
                    })
    
    return sorted(runs, key=lambda x: x['folder'], reverse=True)

# Function to get available case IDs for a specific model run
@st.cache_data
def get_available_case_ids_for_run(run_path):
    if not run_path:
        return []
    
    output_dir = os.path.join(run_path, "output")
    if not os.path.exists(output_dir):
        return []
    
    case_ids = []
    for filename in os.listdir(output_dir):
        if filename.endswith('.txt'):
            try:
                case_id = int(os.path.splitext(filename)[0])
                case_ids.append(case_id)
            except ValueError:
                continue
    
    return sorted(case_ids)

# Function to load model response for a case
@st.cache_data
def load_model_response(run_path, case_id):
    output_file = os.path.join(run_path, "output", f"{case_id}.txt")
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error loading response: {e}"
    return None

# Function to load judge response for a case
@st.cache_data
def load_judge_response(run_path, case_id):
    judge_file = os.path.join(run_path, "judge", f"{case_id}.txt")
    if os.path.exists(judge_file):
        try:
            with open(judge_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error loading judge response: {e}"
    return None

# Function to load scores from detailed.csv
@st.cache_data
def load_detailed_scores(run_path):
    csv_file = os.path.join(run_path, "results", "detailed.csv")
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            return df
        except Exception as e:
            return None
    return None

# Function to get score for specific case and task
def get_case_task_score(scores_df, case_id, task_id):
    if scores_df is None:
        return None
    
    if 'case_id' in scores_df.columns and 'task_id' in scores_df.columns and 'score' in scores_df.columns:
        # Convert to integers for comparison to handle string/int mismatches
        try:
            case_id_int = int(case_id) if case_id is not None else None
            task_id_int = int(task_id) if task_id is not None else None
            
            if case_id_int is not None and task_id_int is not None:
                matching_rows = scores_df[(scores_df['case_id'] == case_id_int) & (scores_df['task_id'] == task_id_int)]
                if not matching_rows.empty:
                    return matching_rows.iloc[0]['score']
        except (ValueError, TypeError):
            # If conversion fails, try string comparison
            matching_rows = scores_df[(scores_df['case_id'] == case_id) & (scores_df['task_id'] == task_id)]
            if not matching_rows.empty:
                return matching_rows.iloc[0]['score']
    
    return None

# Function to parse location from model response
def parse_location_from_response(response_text):
    if not response_text:
        return None
    
    lines = response_text.strip().split('\n')
    lat, lng = None, None
    
    for line in reversed(lines):
        line = line.strip()
        
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

# --- Main App ---

st.set_page_config(layout="wide")

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

query_params = st.query_params
qp_dataset_raw = query_params.get('dataset', None)
if isinstance(qp_dataset_raw, list):
    qp_dataset = qp_dataset_raw[0] if qp_dataset_raw else None
else:
    qp_dataset = qp_dataset_raw
qp_view = query_params.get('view', [None])
qp_case_id = query_params.get('case_id', [None])
qp_model_run = query_params.get('model_run', None)

st.sidebar.header("Navigation")
dataset_base_path = "dataset"
available_datasets = [d for d in os.listdir(dataset_base_path) if os.path.isdir(os.path.join(dataset_base_path, d)) and not d.startswith('.')]

if qp_dataset in available_datasets:
    dataset_default_idx = available_datasets.index(qp_dataset)
else:
    dataset_default_idx = 0
selected_dataset_name = st.sidebar.selectbox("Select Dataset", available_datasets, index=dataset_default_idx)

if selected_dataset_name:
    selected_dataset_path = os.path.join(dataset_base_path, selected_dataset_name)
    metadata = load_metadata(selected_dataset_path)

    if metadata:
        response_runs = get_response_runs(selected_dataset_name)

        if qp_model_run:
            selected_run = next(run for run in response_runs if run['original_name'] == qp_model_run)
        else:
            selected_run = None
        
        if response_runs:
            st.sidebar.header("Model Responses")
            run_options = ["None (Ground Truth Only)"] + [run['display_name'] for run in response_runs]
            selected_run_name = st.sidebar.selectbox("Select Model Run", run_options, index=run_options.index(selected_run['display_name']) if selected_run else 0)
            
            if selected_run_name != "None (Ground Truth Only)":
                selected_run = next(run for run in response_runs if run['display_name'] == selected_run_name)
        
        view_modes = ["Browse Cases", "View All Locations", "Task Distribution", "Response Matrix"]
        if qp_view in view_modes:
            view_default_idx = view_modes.index(qp_view)
        else:
            view_default_idx = 0
        view_mode = st.sidebar.radio(
            "View Mode",
            view_modes,
            index=view_default_idx
        )
        
        if view_mode == "Browse Cases":
            all_case_ids = [case.get("id") for case in metadata.get("cases", []) if case.get("id") is not None]
            
            if selected_run:
                available_case_ids = get_available_case_ids_for_run(selected_run['path'])
                case_ids = [cid for cid in available_case_ids if cid in all_case_ids]
                
                if not case_ids:
                    st.sidebar.warning(f"No cases found for the selected model run.")
                    st.warning(f"The selected model run '{selected_run['display_name']}' has no processed cases.")
            else:
                case_ids = all_case_ids
            
            if not case_ids:
                if not selected_run:
                    st.sidebar.warning("No cases found in the selected dataset's metadata.json.")
                    st.warning(f"No cases (or cases with IDs) found in {selected_dataset_name}/metadata.json")
            else:
                case_id_strs = [str(cid) for cid in sorted(case_ids)]
                if qp_case_id in case_id_strs:
                    case_default_idx = case_id_strs.index(qp_case_id)
                else:
                    case_default_idx = 0
                selected_case_id_str = st.sidebar.radio("Select Case ID", case_id_strs, index=case_default_idx)
                selected_case_id = int(selected_case_id_str)
                
                case_data = get_case_by_id(metadata, selected_case_id)

                if case_data:
                    model_response = None
                    judge_response = None
                    scores_df = None
                    if selected_run:
                        model_response = load_model_response(selected_run['path'], selected_case_id)
                        judge_response = load_judge_response(selected_run['path'], selected_case_id)
                        scores_df = load_detailed_scores(selected_run['path'])
                    
                    col1, spacer, col2 = st.columns([2, 0.15, 2])

                    with col1:
                        if case_data.get("images"):
                            for img_path_suffix in case_data["images"]:
                                full_img_path = os.path.join(selected_dataset_path, img_path_suffix)
                                if os.path.exists(full_img_path):
                                    st.image(full_img_path, caption=os.path.basename(full_img_path), use_container_width=True)
                                else:
                                    st.warning(f"Image not found: {full_img_path}")
                        else:
                            st.write("No images specified for this case.")

                    with col2:
                        st.header(f"Details for Case {selected_case_id}")
                        
                        if case_data.get("info"):
                            st.write(case_data["info"])
                            st.markdown("---")

                        if model_response:
                            st.subheader("🤖 Model Response")
                            with st.expander(f"Response from {selected_run['display_name']}", expanded=True):
                                st.text_area("Raw Response", model_response, height=200)
                            st.markdown("---")

                        if judge_response:
                            st.subheader("⚖️ Judge Response")
                            with st.expander(f"Judge evaluation for {selected_run['display_name']}", expanded=False):
                                st.text_area("Judge Response", judge_response, height=200)
                            st.markdown("---")

                        if case_data.get("tasks"):
                            for i, task in enumerate(case_data["tasks"]):
                                task_id = task.get("id")
                                st.markdown(f"### Task {i+1}")
                                if task.get("type"):
                                    st.markdown(f"**Type:** {task['type']}")
                                if task.get("prompt"):
                                    st.markdown(f"**Prompt:** {task['prompt']}")
                                
                                if scores_df is not None and task_id is not None:
                                    score = get_case_task_score(scores_df, selected_case_id, task_id)
                                    if score is not None:
                                        if isinstance(score, (int, float)):
                                            if score >= 0.8:
                                                st.success(f"**Score:** {score}")
                                            elif score >= 0.5:
                                                st.warning(f"**Score:** {score}")
                                            else:
                                                st.error(f"**Score:** {score}")
                                        else:
                                            st.markdown(f"**Score:** {score}")
                                
                                if task.get("answer") is not None:
                                    st.markdown("**Ground Truth Answer:**")
                                    if isinstance(task["answer"], dict):
                                        st.json(task["answer"])
                                    else:
                                        st.write(task["answer"])
                                if task.get("note"):
                                    st.markdown(f"**Note:** {task['note']}")
                                st.markdown("---")
                        
                    location_tasks = [task for task in case_data.get("tasks", []) if task.get("type") == "location" and task.get("answer")]
                    model_location = None
                    if model_response:
                        model_location = parse_location_from_response(model_response)
                    
                    if location_tasks or model_location:
                        st.markdown("---")
                        st.header("Task Locations")
                        
                        map_points = []
                        
                        for i, task in enumerate(location_tasks):
                            answer = task.get("answer", {})
                            if isinstance(answer, dict) and "lat" in answer and "lng" in answer:
                                try:
                                    lat = float(answer["lat"])
                                    lng = float(answer["lng"])
                                    map_points.append({
                                        "lat": lat,
                                        "lon": lng,
                                        "tooltip": f"Ground Truth - Task {case_data['tasks'].index(task) + 1}",
                                        "color": "red",
                                        "icon": "flag"
                                    })
                                except (ValueError, TypeError):
                                    continue
                        
                        if model_location and selected_run:
                            map_points.append({
                                "lat": model_location["lat"],
                                "lon": model_location["lng"],
                                "tooltip": f"Model Prediction",
                                "color": "blue",
                                "icon": "dot-circle"
                            })
                        
                        if map_points:
                            # Use the average of points as the initial map center
                            avg_lat = sum(p['lat'] for p in map_points) / len(map_points)
                            avg_lon = sum(p['lon'] for p in map_points) / len(map_points)

                            m = folium.Map(
                                location=[avg_lat, avg_lon], 
                                zoom_start=6,
                                scrollWheelZoom=False, 
                                dragging=False,
                                zoom_control=False,
                                tiles=None,
                                attributionControl=False
                            )
                            folium.TileLayer(
                                tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
                                attr=' ',
                                name='CartoDB Dark Matter',
                                control=False
                            ).add_to(m)
                            
                            for point in map_points:
                                folium.Marker(
                                    [point['lat'], point['lon']], 
                                    tooltip=point['tooltip'], 
                                    icon=folium.Icon(color=point['color'], icon=point['icon'], prefix='fa')
                                ).add_to(m)

                            # Auto-fit map to all markers
                            if len(map_points) > 1:
                                bounds = [[p['lat'], p['lon']] for p in map_points]
                                m.fit_bounds(bounds, padding=(30, 30))

                            st_folium(m, use_container_width=True, height=400, returned_objects=[])

                        if len(map_points) >= 2 and any("Ground Truth" in p['tooltip'] for p in map_points) and any("Model Prediction" in p['tooltip'] for p in map_points):
                            ground_truth = next(p for p in map_points if "Ground Truth" in p['tooltip'])
                            model_pred = next(p for p in map_points if "Model Prediction" in p['tooltip'])
                            
                            def haversine_distance(lat1, lon1, lat2, lon2):
                                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                                
                                dlat = lat2 - lat1
                                dlon = lon2 - lon1
                                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                                c = 2 * math.asin(math.sqrt(a))
                                r = 6371
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
            st.header("All Location Tasks")
            
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
                # Use a reasonable default center (world center) and let fit_bounds handle the actual view
                m = folium.Map(location=[20, 0], zoom_start=2, tiles=None, attributionControl=False)
                folium.TileLayer(
                    tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
                    attr=' ',
                    name='CartoDB Dark Matter',
                    control=False
                ).add_to(m)

                for loc in all_locations:
                    folium.Marker(
                        [loc['lat'], loc['lon']],
                        icon=folium.Icon(color='blue', icon='dot-circle', prefix='fa'),
                        tooltip=f"Case {loc['case_id']}, Task {loc['task_num']}"
                    ).add_to(m)

                # Auto-fit map to all markers
                bounds = [[loc['lat'], loc['lon']] for loc in all_locations]
                m.fit_bounds(bounds, padding=(30, 30))

                # This map is for exploration, so we keep it interactive
                st_folium(m, use_container_width=True, height=500)
                
                st.markdown("---")
                st.subheader("Location Details")
                
                for i, loc in enumerate(all_locations):
                    with st.expander(f"Case {loc['case_id']}, Task {loc['task_num']} - {loc['lat']:.6f}, {loc['lon']:.6f}"):
                        st.write(f"**Task Prompt:** {loc['prompt']}")
                        st.write(f"**Case Info:** {loc['info']}")
                        st.write(f"**Coordinates:** {loc['lat']}, {loc['lon']}")
            else:
                st.info(f"No location tasks found in the {selected_dataset_name} dataset.")
        
        elif view_mode == "Response Matrix":
            st.header("Response Matrix")
            if not response_runs:
                st.info("No model runs found for this dataset.")
            else:
                # Group tasks by case
                case_task_groups = {}
                for case in metadata.get("cases", []):
                    case_id = case.get("id")
                    if case_id is not None:
                        case_task_groups[case_id] = []
                        for task_idx, task in enumerate(case.get("tasks", [])):
                            task_id = task.get("id")
                            case_task_groups[case_id].append({
                                "task_id": task_id,
                                "task_idx": task_idx
                            })
                
                run_data = []
                for run in response_runs:
                    scores_df = load_detailed_scores(run['path'])
                    run_data.append({
                        "run": run,
                        "scores_df": scores_df,
                    })
                
                html = "<table style='border-collapse:collapse;width:100%;'>"
                html += "<tr><th style='border:1px solid #ccc;padding:4px;'>Case</th><th style='border:1px solid #ccc;padding:4px;'>Task</th>"
                for run in response_runs:
                    html += f"<th style='border:1px solid #ccc;padding:4px;' title='{run['original_name']}'>" + run['display_name'] + "</th>"
                html += "</tr>"
                
                for case_id in sorted(case_task_groups.keys()):
                    tasks = case_task_groups[case_id]
                    for i, task in enumerate(tasks):
                        html += "<tr>"
                        
                        # Case column - only show on first task of each case
                        if i == 0:
                            rowspan = len(tasks)
                            html += f"<td style='border:1px solid #ccc;padding:4px;' rowspan='{rowspan}'>C{case_id}</td>"
                        
                        # Task column
                        html += f"<td style='border:1px solid #ccc;padding:4px;'>{task['task_idx']+1}</td>"
                        
                        # Score columns for each model run
                        for run_idx, run in enumerate(response_runs):
                            scores_df = run_data[run_idx]['scores_df']
                            score = get_case_task_score(scores_df, case_id, task['task_id']) if scores_df is not None else None
                            if score is None:
                                color = '#eee'
                            elif isinstance(score, (int, float)):
                                if score >= 0.8:
                                    color = '#b6fcb6'
                                elif score >= 0.5:
                                    color = '#fff7b2'
                                else:
                                    color = '#ffb2b2'
                            else:
                                color = '#eee'
                            
                            # Create clickable link for the score
                            case_link = f"?dataset={selected_dataset_name}&view=Browse%20Cases&case_id={case_id}&model_run={run['original_name']}"
                            score_disp = f"<a href='{case_link}' style='color:#222;font-weight:bold;text-decoration:none;'>{score:.2f}</a>" if isinstance(score, (int, float)) else ""
                            html += f"<td style='border:1px solid #ccc;padding:4px;background:{color};vertical-align:top;'>{score_disp}</td>"
                        
                        html += "</tr>"
                
                html += "</table>"
                st.markdown(html, unsafe_allow_html=True)
        else: # Task Distribution
            st.header("📊 Task Type Distribution")
            
            task_counts = {}
            total_tasks = 0
            total_cases = len(metadata.get("cases", []))
            
            for case in metadata.get("cases", []):
                case_tasks = case.get("tasks", [])
                total_tasks += len(case_tasks)
                
                for task in case_tasks:
                    task_type = task.get("type", "unknown")
                    task_counts[task_type] = task_counts.get(task_type, 0) + 1
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cases", total_cases)
            with col2:
                st.metric("Total Tasks", total_tasks)
            with col3:
                st.metric("Task Types", len(task_counts))
            
            if task_counts:
                st.markdown("---")
                
                st.subheader("Task Type Distribution")
                
                df = pd.DataFrame(list(task_counts.items()), columns=['Task Type', 'Count'])
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.pie(df['Count'], labels=df['Task Type'], autopct='%1.1f%%')
                    ax.axis('equal')
                    st.pyplot(fig)
                
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