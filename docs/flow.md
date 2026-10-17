flowchart TD
    Start([User Query]) --> Rewriter[Node 1: Query Writer / Rewriter]
    Rewriter --> InfoCheck{Node 2: Need More Info?}
    
    %% Branch: No Info Needed
    InfoCheck -- No --> GenDirect[Node 4: DeepSeek Generator Direct]
    
    %% Branch: Info Needed
    InfoCheck -- Yes --> SourceRouter[Node 3: Source Selector Agent]
    
    subgraph KnowledgeRetrieval [Multi-Source Ingestion]
        SourceRouter --> VDB[(Vector DB)]
        SourceRouter --> Web[Internet Search]
        SourceRouter --> Tools[Tools & APIs]
    end
    
    VDB --> ContextMerge[Context Fusion & Reranker]
    Web --> ContextMerge
    Tools --> ContextMerge
    
    ContextMerge --> GenRAG[Node 4: DeepSeek Generator with Context]
    
    GenDirect --> Eval[Node 5: Response Relevance & Hallucination Evaluator]
    GenRAG --> Eval
    
    Eval --> IsRelevant{Is Response Relevant & Grounded?}
    
    IsRelevant -- Yes --> Final([Final Response to User])
    IsRelevant -- "No (Critique & Retry)" --> RetryCheck{Max Retries Exceeded?}
    RetryCheck -- No --> Rewriter
    RetryCheck -- Yes --> Fallback([Graceful Fallback Response])
