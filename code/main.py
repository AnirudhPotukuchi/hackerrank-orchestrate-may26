import pandas as pd
from tqdm import tqdm
from agent import Agent
from config import INPUT_CSV, OUTPUT_CSV

def main():
        
    print(f"Loading support tickets from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_CSV}")
        return

    agent = Agent()
    
    # Ingest data first if not already ingested
    print("Ensuring data is ingested in ChromaDB...")
    agent.retriever.ingest_data()
    
    results = []
    
    print("Processing tickets...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Handle potential NaNs by checking the capitalized CSV column names
        issue = str(row['Issue']) if pd.notna(row.get('Issue')) else ""
        subject = str(row['Subject']) if pd.notna(row.get('Subject')) else ""
        company = str(row['Company']) if pd.notna(row.get('Company')) else "None"
        
        result_dict = agent.process_ticket(issue, subject, company)
        
        # Combine input and output in the exactly requested column order
        combined = {
            "issue": issue,
            "subject": subject,
            "company": company,
            "response": result_dict.get("response", ""),
            "product_area": result_dict.get("product_area", "unknown"),
            "status": result_dict.get("status", "escalated"),
            "request_type": result_dict.get("request_type", "invalid"),
            "justification": result_dict.get("justification", ""),
        }
        results.append(combined)
        
    output_df = pd.DataFrame(results)
    
    print(f"Saving results to {OUTPUT_CSV}...")
    output_df.to_csv(OUTPUT_CSV, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
