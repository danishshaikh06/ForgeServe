Title: Latency-Aware Continuous Batching

Description: Reducing request queueing/wait time while maintaining batched decoding.

Step 1: [A B C]
Step 2: [A B C D]    ← D joins
Step 3: [A C D]      ← B finishes
Step 4: [A C D E]    ← E joins
Step 5: [A D E]      ← C finishes
