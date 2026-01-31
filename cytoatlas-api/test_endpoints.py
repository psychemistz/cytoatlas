#!/usr/bin/env python3
"""Test the new CytoAtlas endpoints."""

import json
import subprocess
import time
import sys
import os

# Disable proxy
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ['no_proxy'] = 'localhost,127.0.0.1'

import requests

BASE_URL = "http://127.0.0.1:8766/api/v1"

def test_health():
    print("=== Testing Health ===")
    r = requests.get(f"{BASE_URL}/health")
    print(json.dumps(r.json(), indent=2))
    print()

def test_search():
    print("=== Testing Search (IFNG) ===")
    r = requests.get(f"{BASE_URL}/search", params={"q": "IFNG", "limit": 5})
    data = r.json()
    print(f"Total results: {data.get('total_results', 0)}")
    for result in data.get('results', [])[:5]:
        print(f"  - {result['type']}: {result['name']} ({result['atlas_count']} atlases)")
    print()

def test_autocomplete():
    print("=== Testing Autocomplete (IL) ===")
    r = requests.get(f"{BASE_URL}/search/autocomplete", params={"q": "IL", "limit": 5})
    data = r.json()
    for s in data.get('suggestions', []):
        print(f"  - {s['type']}: {s['text']}")
    print()

def test_search_stats():
    print("=== Testing Search Stats ===")
    r = requests.get(f"{BASE_URL}/search/stats")
    data = r.json()
    print(f"Total entities: {data.get('total_entities', 0)}")
    print(f"By type: {data.get('by_type', {})}")
    print()

def test_chat_status():
    print("=== Testing Chat Status ===")
    r = requests.get(f"{BASE_URL}/chat/status")
    print(json.dumps(r.json(), indent=2))
    print()

def test_chat_suggestions():
    print("=== Testing Chat Suggestions ===")
    r = requests.get(f"{BASE_URL}/chat/suggestions")
    data = r.json()
    for s in data.get('suggestions', [])[:4]:
        print(f"  [{s['category']}] {s['text']}")
    print()

def test_chat_message():
    print("=== Testing Chat Message ===")
    r = requests.post(
        f"{BASE_URL}/chat/message",
        json={
            "content": "What is the activity of IFNG in CD8 T cells? Use the search tool.",
            "session_id": "test123"
        }
    )
    if r.status_code == 200:
        data = r.json()
        print(f"Message ID: {data.get('message_id')}")
        print(f"Conversation ID: {data.get('conversation_id')}")
        print(f"Content (truncated): {data.get('content', '')[:500]}...")
        if data.get('tool_calls'):
            print(f"Tools used: {[t['name'] for t in data['tool_calls']]}")
        if data.get('visualizations'):
            print(f"Visualizations: {len(data['visualizations'])}")
    else:
        print(f"Error: {r.status_code}")
        print(r.text[:500])
    print()

def test_submit_types():
    print("=== Testing Submit Signature Types ===")
    r = requests.get(f"{BASE_URL}/submit/signature-types")
    print(json.dumps(r.json(), indent=2))
    print()

def test_websocket_status():
    print("=== Testing WebSocket Status ===")
    r = requests.get(f"{BASE_URL}/ws/status")
    print(json.dumps(r.json(), indent=2))
    print()

if __name__ == "__main__":
    try:
        test_health()
        test_search()
        test_autocomplete()
        test_search_stats()
        test_chat_status()
        test_chat_suggestions()
        test_submit_types()
        test_websocket_status()

        if len(sys.argv) > 1 and sys.argv[1] == "--chat":
            test_chat_message()
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        print("Make sure the server is running on port 8766")
