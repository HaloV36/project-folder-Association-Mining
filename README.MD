### Interactive Supermarket Simulation with Association Rule Mining

#### Author Information

- **Name**: Alexander Alfonso
- **Student ID**: 6403226
- **Course**: CAI 4002 - Artificial Intelligence
- **Semester**: Fall



#### System Overview

The application serves as a simulation for a supermarket shopping experience and does association rule mining to discover different purchasing patterns. (Xian Su). You’re able to select various products and save them as a transaction. After that you can use preprocessing and either apriori or eclat algorithm, to determine the likely hood that customers would who bought one item, would buy another item.



#### Technical Stack

- **Language**: Python 3.x

- **Key Libraries**: streamlit, pandas, numpy

- **UI Framework**: Streamlit (brower based)



#### Installation

-movde form don

##### Prerequisites
- Python 3.9+
- pip

##### Setup
```bash
# Clone or extract project
cd [project-directory]

# Install dependencies
[command to install dependencies]
pip install streamlit pandas numpy

# Run application
[command to start application]
python -m streamlit run src/main.py
```



#### Usage

##### 1. Load Data
- **Manual Entry**: Click items to create transactions
                  - After the transactions get picked, make sure to press "Save Transaction" to save the entries.
                  - Manually added Transactions will appear below in a chart.
- **Import CSV**: Use "Import" button to load `sample_transactions.csv` or use default sample_transactions.csv

##### 2. Preprocess Data
- Click "Run Preprocessing"
- Review cleaning report (empty transactions, duplicates, etc.)
- Scroll down to see a chart that will also show all cleaned transactions

##### 3. Run Mining
- Set minimum support and confidence thresholds
    - Minimum Support default = 0.2
    - Minimum Confidence default = 0.5
- Click "Run Apriori" to execute Apriori
- Click "Run Eclat" to execute Eclat
- Wait for completion (~1-3 seconds)
- Chart below shows frequent itemsets and rules generated for Apriori and Eclat
- Click on show detailed Rules, and choose either Apriori or Eclat to observe rules.

##### 4. Query Results
- Select the algorithm for recommendation
- Select product from dropdown
- View associated items and recommendation strength



#### Algorithm Implementation

##### Apriori
[2-3 sentences on your implementation approach]
- Data structure: [e.g., dictionary of itemsets]
- Candidate generation: [breadth-first, level-wise]
- Pruning strategy: [minimum support]

##### Eclat
[2-3 sentences on your implementation approach]
- Data structure: [e.g., TID-set representation]
- Search strategy: [depth-first]
- Intersection method: [set operations]



#### Performance Results

Tested on provided dataset (80-100 transactions after cleaning):

| Algorithm | Runtime (ms) | Rules Generated | Memory Usage |
|-----------|--------------|-----------------|--------------|
| Apriori   | [value]      | [value]         | [value]      |
| Eclat     | [value]      | [value]         | [value]      |

**Parameters**: min_support = 0.2, min_confidence = 0.5

**Analysis**: [1-2 sentences explaining performance differences]



#### Project Structure

```
project-root/
├── src/
│   ├── algorithms/
│   │   ├── apriori.[py/js/java]
│   │   ├── eclat.[py/js/java]
│   │   └──
│   ├── preprocessing/
│   │   └── cleaner.[py/js/java]
│   ├── ui/
│   │   └── [interface files]
│   └── main.[py/js/java]
├── data/
│   ├── sample_transactions.csv
│   └── products.csv
├── README.md
├── REPORT.pdf
└── [requirements.txt / package.json / pom.xml]
```



#### Data Preprocessing

Preprocessing Report for default sample_transactions.csv:
---------------------
Before Cleaning:
- Total transactions: 95
- Empty transactions (or became empty): 0
- Single-item transactions: 6
- Invalid items removed: 2

After Cleaning:
- Valid transactions: 89
- Total items (after cleaning): 276
- Unique products: 30
- Transactions removed: 6



#### Testing

Verified functionality:
- [✓] CSV import and parsing
- [✓] All preprocessing operations
- [✓] Three algorithm implementations
- [✓] Interactive query system
- [✓] Performance measurement

Test cases:
- Test case 1: Using default sample_transactions with no additions and a minimum support of 0.2 and a minimum confidence of 0.5 we get
    - 7 frequent itemsets  
    - 11 rules generated 
- Test case 2: Using default sample_transactions with no additions and a minimum support of 0.1 and a minimum confidence of 0.5 we get
    - 25





#### Known Limitations

-Biggest limiation is that the executable run time is rounded up alot. So it will show it took 0ms to run but really it just means it took less than 1 ms to run the algorithm.


#### AI Tool Usage

[Required: 1 paragraph describing which AI tools you used and for what purpose]

Used ChatGPT, to write all the code to run the supermarket shopping application. Used chatGPT various times to fix any errors that the code would run into, and also to explain how the code works and functions. I also used chatgpt to explain he implementaion of both the Apriori and Eclat algorithm. ChatGPT has helped me fill out various things within the readme and report.

Example:
"Used ChatGPT for explaining Eclat algorithm vertical representation and debugging file parsing errors. Used GitHub Copilot for generating UI boilerplate code. All generated code was reviewed, tested, and adapted for this specific implementation."



#### References

- Course lecture materials
- [Algorithm papers or resources consulted]
- [Library documentation links]