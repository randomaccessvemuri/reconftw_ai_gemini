#!/usr/bin/env python3

import os
import argparse
import glob
import sys
import json
from datetime import datetime
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables (useful for API_KEY)
load_dotenv()

# Default configuration
DEFAULT_RECONFTW_RESULTS_DIR = "./reconftw_output"
DEFAULT_OUTPUT_DIR = "./reconftw_ai_output"
DEFAULT_MODEL_NAME = "gemini-2.0-flash"  # Updated for Gemini
DEFAULT_OUTPUT_FORMAT = "txt"
DEFAULT_REPORT_TYPE = "executive"
DEFAULT_PROMPTS_FILE = "prompts.json"

REPORT_TYPES = ["executive", "brief", "bughunter"]
OUTPUT_FORMATS = ["txt", "md"]
CATEGORIES = ["osint", "subdomains", "hosts", "webs"]

def load_prompts(prompts_file: str) -> Dict:
    try:
        with open(prompts_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Prompts file '{prompts_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in prompts file: {e}")
        sys.exit(1)

def get_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY not found in environment variables.")
        sys.exit(1)
    return genai.Client(api_key=api_key)

def read_files(category: str, results_dir: str) -> str:
    combined_data = ""
    category_dir = os.path.join(results_dir, category)

    if not os.path.isdir(category_dir):
        return f"[Error] Directory {category_dir} does not exist."

    file_paths = glob.glob(os.path.join(category_dir, "**/*"), recursive=True)

    for file_path in file_paths:
        if os.path.isfile(file_path):
            relative_path = os.path.relpath(file_path, results_dir)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    combined_data += f"--- {relative_path} ---\n{f.read().strip()}\n"
            except Exception as e:
                combined_data += f"[Error] Failed to read {relative_path}: {str(e)}\n"

    if not combined_data:
        return f"[Info] No files found in {category_dir}."

    return combined_data.strip()

def process_category(client, category: str, data: str, model_name: str, report_type: str, base_prompts: Dict) -> str:
    if not data or "[Error]" in data:
        return f"[Error] No valid data available for {category}."

    prompt_template = base_prompts.get(report_type, {}).get(category, "Analyze this reconnaissance data and highlight high-risk findings:\n{data}")
    prompt = prompt_template.format(data=data)

    try:
        # Gemini API call
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"[Error] Failed to process {category} with Gemini: {str(e)}"

def save_results(results: Dict[str, str], output_dir: str, model_name: str, output_format: str, report_type: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = "md" if output_format == "md" else "txt"
    output_file = os.path.join(output_dir, f"reconftw_analysis_{report_type}_{timestamp}.{extension}")

    with open(output_file, "w", encoding="utf-8") as f:
        if output_format == "md":
            f.write(f"# ReconFTW-AI Analysis (Gemini)\n\n")
            f.write(f"- **Model Used**: `{model_name}`\n")
            f.write(f"- **Report Type**: `{report_type}`\n")
            f.write(f"- **Date**: `{timestamp}`\n\n")
            for category, interpretation in results.items():
                f.write(f"## {category.upper()}\n\n{interpretation}\n\n")
        else:
            f.write(f"ReconFTW-AI Analysis\nModel: {model_name}\nReport Type: {report_type}\nDate: {timestamp}\n")
            f.write("=" * 60 + "\n\n")
            for category, interpretation in results.items():
                f.write(f"=== {category.upper()} ===\n{interpretation}\n\n")

    print(f"[*] Results saved to '{output_file}'")

def analyze_reconftw_results(client, results_dir: str, model_name: str, report_type: str, base_prompts: Dict) -> Dict[str, str]:
    results = {}
    
    # Step 1: Read all files in parallel
    raw_data_per_category = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(read_files, cat, results_dir): cat for cat in CATEGORIES}
        for future in as_completed(futures):
            raw_data_per_category[futures[future]] = future.result()

    # Step 2: Process categories with Gemini
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_category, client, cat, raw_data_per_category[cat], model_name, report_type, base_prompts): cat 
            for cat in CATEGORIES
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    # Step 3: Global Overview
    all_data = "\n\n".join([f"{k.upper()}:\n{v}" for k, v in raw_data_per_category.items()])
    results["overview"] = process_category(client, "overview", all_data, model_name, report_type, base_prompts)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="ReconFTW-AI: Use Gemini to interpret ReconFTW results")
    parser.add_argument("--results-dir", default=DEFAULT_RECONFTW_RESULTS_DIR, help="Directory with ReconFTW results.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where to save the analysis.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Gemini model (e.g., gemini-2.0-flash or gemini-1.5-pro).")
    parser.add_argument("--output-format", choices=OUTPUT_FORMATS, default=DEFAULT_OUTPUT_FORMAT, help="Output format: txt or md.")
    parser.add_argument("--report-type", choices=REPORT_TYPES, default=DEFAULT_REPORT_TYPE, help="Type of report to generate.")
    parser.add_argument("--prompts-file", default=DEFAULT_PROMPTS_FILE, help="JSON file containing prompt templates.")

    args = parser.parse_args()

    # Initialization
    base_prompts = load_prompts(args.prompts_file)
    client = get_gemini_client()

    print(f"[*] Analyzing ReconFTW results with {args.model}...")
    results = analyze_reconftw_results(client, args.results_dir, args.model, args.report_type, base_prompts)

    save_results(results, args.output_dir, args.model, args.output_format, args.report_type)

if __name__ == "__main__":
    main()