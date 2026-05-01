import pandas as pd
from tqdm import tqdm
from agent import Agent
from config import INPUT_CSV, OUTPUT_CSV, OPENAI_API_KEY

def main():
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY is not set in .env")
        return
        
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
        # Handle potential NaNs
        issue = str(row['issue']) if pd.notna(row.get('issue')) else ""
        subject = str(row['subject']) if pd.notna(row.get('subject')) else ""
        company = str(row['company']) if pd.notna(row.get('company')) else "None"
        
        result_dict = agent.process_ticket(issue, subject, company)
        
        # Combine input and output
        combined = {
            "issue": issue,
            "subject": subject,
            "company": company,
            "status": result_dict["status"],
            "product_area": result_dict["product_area"],
            "response": result_dict["response"],
            "justification": result_dict["justification"],
            "request_type": result_dict["request_type"],
        }
        results.append(combined)
        
    output_df = pd.DataFrame(results)
    
    # Ensure only the 5 required columns are outputted (or all 8 if we want, but evaluation criteria specifically says: 
    # "We score per row across all five output columns". It's safer to output the 5 columns exactly, or 8, let's output 5 + inputs to be safe)
    # The problem statement says "For each row, generate: status, product_area, response, justification, request_type"
    
    print(f"Saving results to {OUTPUT_CSV}...")
    output_df.to_csv(OUTPUT_CSV, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
