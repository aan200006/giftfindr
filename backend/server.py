from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from semantic_search import semantic_search
import openai
from openai import OpenAI
import os
import re
import json

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

app = Flask(__name__)
CORS(app)

# openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))


@app.route("/api/search", methods=["POST"])
def search():
    data = request.json

    k = int(data.get("k", 5))
    query = "{recipient} gift, {occation}, {interests}"
    if not query:
        print("Missing 'query' in request data")
        return jsonify({"error": "Missing 'query'"}), 400
    results = semantic_search(query, k)

    # using price filter over under results
    price = data.get("price")
    offset = 40
    if (
        price
        and isinstance(price, (int, float))
        or (isinstance(price, str) and price.replace(".", "", 1).isdigit())
    ):
        price = float(price)  # Ensure price is a float for comparison
        results = [
            result for result in results if float(result["price"]) <= price + offset
        ]
    return jsonify(results)


@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    try:
        messages = request.json.get("messages", [])
        system_prompt = {
            "role": "system",
            "content": 'Ask for things like "recipient", "name", "age","interests", "budget", "occasion"',
        }
        conv = [system_prompt] + messages
        response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=conv)
        aiMessage = response.choices[0].message.content
        return jsonify({"message": aiMessage})
    except Exception as e:
        print("Error fetching from OpenAI:", e)
        return (
            jsonify({"error": "Something went wrong while processing the request."}),
            500,
        )


@app.route("/api/chatbot/json", methods=["POST"])
def chatbot_json():
    try:
        messages = request.json.get("messages", [])
        previousData = request.json.get("previousStructuredData", {})
        systemMessage = {
            "role": "system",
            "content": 'Extract the gift preferences and format as a structured JSON with keys "recipient", "name", "age", "interests", "min_budget","max_budget", "occasion". If not applicable, set to N/A. Additionally, provide a natural response to continue the conversation.',
        }
        conv = [systemMessage] + messages
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conv,
        )

        aiMessage = response.choices[0].message.content

        jsonMatch = re.search(r"{.*}", aiMessage, re.DOTALL)
        print(aiMessage)
        structured = None
        if jsonMatch:
            try:
                structured = json.loads(jsonMatch.group(0))
                print(structured)
            except json.JSONDecodeError:
                print("Failed to parse JSON from AI response:", aiMessage)
                structured = {
                    "error": "Unable to parse structured response:",
                    "message": aiMessage,
                }

        if previousData:
            print(previousData)
            structured = {**previousData, **(structured or {})}
        return jsonify({"message": aiMessage, "structured": structured})
    except Exception as e:
        print("Error fetching from OpenAI:", e)
        return (
            jsonify({"error": "Something went wrong while processing the request."}),
            500,
        )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
