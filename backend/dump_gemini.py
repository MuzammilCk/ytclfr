import os
from pymongo import MongoClient

client = MongoClient("mongodb://mongo:27017")
db = client["ytclassifier_docs"]
doc = db.analysis_results.find_one({"_analysis_id": "eae0466a-8444-473a-a696-f00dc1692943"})
if doc and "classification" in doc and "raw_response" in doc["classification"]:
    with open("raw_gemini_resp.txt", "w", encoding="utf-8") as f:
        f.write(doc["classification"]["raw_response"])
    print("Dumped raw response.")
else:
    print("No raw_response found in DB document.")
