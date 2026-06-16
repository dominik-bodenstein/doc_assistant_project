
# Project WriteUp

## Architecture

Classify Intent llm splits user intent into 3 paths based on content, so each path can have different agents with different prompting and workflow

- Q&A tasks focus on retrieval and direct answers.
- Summarization tasks focus on compression and key points.
- Calculation tasks focus on numeric reasoning and tool-assisted math.

After Each Path a update_memory agent is invoked to summarize the message history

### State & Memory

AgentState is Saved to AgentState typed Dictfor each run
Ephemeral memory is handled via Session id and SessionState updated after each turn
For Persitance Sessions can be saved and loaded by session id

### Type Checking

States types are enforced by TypedDicts and tyechecking.
Llm results are enforced by Schemas, if shema is not conformed, an error is risen.

### Tooling decisions

Tools are registered once in `DocumentAssistant` and passed in workflow config:

- `calculator`
- `document_search`
- `document_reader`
- `document_statistics`

### Graph Network

```text
                                                +-----------+
                                                | __start__ |
                                                +-----------+
                                                      *
                                                      *
                                                      *
                                            +-----------------+
                                            | classify_intent |..
                                     .......+-----------------+  ........
                             ........        ...           ...           .......
                      .......              ..                 .                 .......
                  ....                   ..                    ..                      ........
+-------------------+           +----------+           +---------------------+                 ....
| calculation_agent |           | qa_agent |           | summarization_agent |                    .
+-------------------+**         +----------+           +---------------------+                    .
                       ****           *            ****                                           .
                           *****      *       *****                                               .
                                ***   *    ***                                                    .
                              +---------------+                                                ....
                              | update_memory |                                        ........
                              +---------------+                                ........
                                             ***                       ........
                                                **             ........
                                                  **       ....
                                                +---------+
                                                | __end__ |
                                                +---------+
```

## Example Chats

```text
*********************************************************************************
Enter Message: please get  the total sum of all the invoces

Processing...

Assistant: Returning structured response: expression='20000 + 69300 + 214500' result=303800.0 explanation='The total sum of all invoices is calculated by adding the amounts from each invoice: $20,000 from INV-001, $69,300 from INV-002, and $214,500 from INV-003.' units='USD' timestamp=datetime.datetime(2026, 6, 15, 19, 41, 30, 182884)

INTENT: calculation

TOOLS USED: document_statistics, document_search, document_reader, document_reader, document_reader, calculator, calculator

CONVERSATION SUMMARY: The total sum of all invoices is $303,800. This was calculated by adding the amounts from three invoices: $20,000 from Invoice #12345, $69,300 from Invoice #12346, and $214,500 from Invoice #12347.

*********************************************************************************

Enter Message: pelase summarize inv 1

Processing...

Assistant: Returning structured response: original_length=233 summary='Invoice #12345 issued to Acme Corporation on January 15, 2024, includes charges for consulting services ($5,000), software development ($12,500), and support & maintenance ($2,500). The subtotal is $20,000 with a 10% tax of $2,000, totaling $22,000. Payment is due within 30 days.' key_points=['Invoice #12345', 'Date: 2024-01-15', 'Client: Acme Corporation', 'Consulting Services: $5,000', 'Software Development: $12,500', 'Support & Maintenance: $2,500', 'Subtotal: $20,000', 'Tax (10%): $2,000', 'Total: $22,000', 'Payment Terms: Net 30 days'] document_ids=['INV-001'] timestamp=datetime.datetime(2024, 1, 15, 0, 0, tzinfo=TzInfo(0))

INTENT: summarization

TOOLS USED: document_reader

CONVERSATION SUMMARY: Invoice #12345 issued to Acme Corporation on January 15, 2024, includes charges for consulting services ($5,000), software development ($12,500), and support & maintenance ($2,500). The subtotal is $20,000 with a 10% tax of $2,000, totaling $22,000. Payment is due within 30 days.

*********************************************************************************

Enter Message: when i payment due?

Processing...

Assistant: Returning structured response: question='when i payment due?' answer='Here are the payment due dates for the invoices found:\n\n1. **Invoice #12345** (Client: Acme Corporation)\n   - **Date:** 2024-01-15\n   - **Payment Terms:** Net 30 days\n   - **Due Date:** 2024-02-14\n\n2. **Invoice #12346** (Client: TechStart Inc.)\n   - **Date:** 2024-02-20\n   - **Payment Terms:** Net 45 days\n   - **Due Date:** 2024-04-05\n\n3. **Invoice #12347** (Client: Global Corp)\n   - **Date:** 2024-03-01\n   - **Payment Terms:** Net 60 days\n   - **Due Date:** 2024-04-30' sources=['INV-001', 'INV-002', 'INV-003'] confidence=0.95 timestamp=datetime.datetime(2026, 6, 15, 19, 45, 1, 266467)

INTENT: qa

TOOLS USED: document_search, document_reader, document_reader, document_reader

CONVERSATION SUMMARY: The conversation involved checking the payment due dates for three invoices. Invoice #12345 for Acme Corporation is due on 2024-02-14, Invoice #12346 for TechStart Inc. is due on 2024-04-05, and Invoice #12347 for Global Corp is due on 2024-04-30.
*********************************************************************************
```
