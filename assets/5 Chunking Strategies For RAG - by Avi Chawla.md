[![Daily Dose of Data Science](https://substackcdn.com/image/fetch/$s_!heKx!,w_80,h_80,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc5dc1fee-2d1e-4892-b219-4b96f6998ab5_288x288.png)](https://blog.dailydoseofds.com/)

# [Daily Dose of Data Science](https://blog.dailydoseofds.com/)

SubscribeSign in

![User's avatar](https://substackcdn.com/image/fetch/$s_!fRqh!,w_64,h_64,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa0dc0dc6-c4ff-4fe7-b467-bfb654e7dc6f_287x287.jpeg)

Discover more from Daily Dose of Data Science

A free newsletter for continuous learning about data science and ML, lesser-known techniques, and how to apply them in 2 minutes. We keep things no-fluff.

Join 100,000+ data scientists from top companies like Google, NVIDIA, Microsoft, Uber, etc.

Subscribe

By subscribing, I agree to Substack's [Terms of Use](https://substack.com/tos), and acknowledge its [Information Collection Notice](https://substack.com/ccpa#personal-data-collected) and [Privacy Policy](https://substack.com/privacy).

Already have an account? Sign in

# 5 Chunking Strategies For RAG

### ...explained in a single frame.

[![Avi Chawla's avatar](https://substackcdn.com/image/fetch/$s_!fRqh!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa0dc0dc6-c4ff-4fe7-b467-bfb654e7dc6f_287x287.jpeg)](https://substack.com/@avichawla)

[Avi Chawla](https://substack.com/@avichawla)

Oct 18, 2024

130

[5](https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag/comments)
9

Share

Here’s the typical workflow of a RAG application:

[![](https://substackcdn.com/image/fetch/$s_!DpHM!,w_1456,c_limit,f_auto,q_auto:good,fl_lossy/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6878b8fa-5e74-45a1-9a89-5aab92889126_2366x990.gif)](https://substackcdn.com/image/fetch/$s_!DpHM!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6878b8fa-5e74-45a1-9a89-5aab92889126_2366x990.gif) RAG: Store additional information as vectors, match the incoming query to those vectors, and feed the most similar info to the LLM along with the query.

Since the additional document(s) can be pretty large, step 1 also involves chunking, wherein a large document is divided into smaller/manageable pieces.

[![](https://substackcdn.com/image/fetch/$s_!RDxE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdeab4ef3-d3ec-4459-8004-ceffe81652ca_1829x392.png)](https://substackcdn.com/image/fetch/$s_!RDxE!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdeab4ef3-d3ec-4459-8004-ceffe81652ca_1829x392.png)

This step is crucial since it ensures the text fits the input size of the embedding model.

Moreover, it enhances the efficiency and accuracy of the retrieval step, which directly impacts the quality of generated responses ( [we discussed this yesterday](https://www.dailydoseofds.com/bi-encoders-and-cross-encoders-for-sentence-pair-similarity-scoring-part-1/)).

Here are five chunking strategies for RAG:

[![](https://substackcdn.com/image/fetch/$s_!us2E!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F92c70184-ba0f-4877-9a55-e4add0e311ad_870x1116.gif)](https://substackcdn.com/image/fetch/$s_!us2E!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F92c70184-ba0f-4877-9a55-e4add0e311ad_870x1116.gif)

Let’s understand them today!

> _Note: Yesterday, we discussed techniques to build robust NLP systems that rely on pairwise content similarity (RAG is one of them). Read here in case you missed it: **[Bi-encoders and Cross-encoders for Sentence Pair Similarity Scoring – Part 1](https://www.dailydoseofds.com/bi-encoders-and-cross-encoders-for-sentence-pair-similarity-scoring-part-1/)**._

* * *

### 1) Fixed-size chunking

The most intuitive and straightforward way to generate chunks is by splitting the text into uniform segments based on a pre-defined number of characters, words, or tokens.

[![](https://substackcdn.com/image/fetch/$s_!RG5y!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F98c422a0-f0e2-457c-a256-4476a56a601f_943x232.png)](https://substackcdn.com/image/fetch/$s_!RG5y!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F98c422a0-f0e2-457c-a256-4476a56a601f_943x232.png)

Since a direct split can disrupt the semantic flow, it is recommended to maintain some overlap between two consecutive chunks (the blue part above).

This is simple to implement. Also, since all chunks are of equal size, it simplifies batch processing.

But there is a big problem. This usually breaks sentences (or ideas) in between. Thus, important information will likely get distributed between chunks.

* * *

### 2) Semantic chunking

The idea is simple.

[![](https://substackcdn.com/image/fetch/$s_!tmOD!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa6ad83a6-2879-4c77-9e49-393f16577aef_1066x288.gif)](https://substackcdn.com/image/fetch/$s_!tmOD!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa6ad83a6-2879-4c77-9e49-393f16577aef_1066x288.gif)

- Segment the document based on meaningful units like sentences, paragraphs, or thematic sections.

- Next, create embeddings for each segment.

- Let’s say I start with the first segment and its embedding.

  - If the first segment’s embedding has a high cosine similarity with that of the second segment, both segments form a chunk.

  - This continues until cosine similarity drops significantly.

  - The moment it does, we start a new chunk and repeat.

Here’s what the output could look like:

[![](https://substackcdn.com/image/fetch/$s_!sTc2!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F74037e11-362d-4ea2-8ee2-ee85ab013523_963x231.png)](https://substackcdn.com/image/fetch/$s_!sTc2!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F74037e11-362d-4ea2-8ee2-ee85ab013523_963x231.png)

Unlike fixed-size chunks, this maintains the natural flow of language and preserves complete ideas.

Since each chunk is richer, it improves the retrieval accuracy, which, in turn, produces more coherent and relevant responses by the LLM.

A minor problem is that it depends on a threshold to determine if cosine similarity has dropped significantly, which can vary from document to document.

* * *

### 3) Recursive chunking

This is also simple.

[![](https://substackcdn.com/image/fetch/$s_!WRuN!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff4009caa-34fc-48d6-8102-3d0f6f2c1386_1066x316.gif)](https://substackcdn.com/image/fetch/$s_!WRuN!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff4009caa-34fc-48d6-8102-3d0f6f2c1386_1066x316.gif)

First, chunk based on inherent separators like paragraphs, or sections.

Next, split each chunk into smaller chunks if the size exceeds a pre-defined chunk size limit. If, however, the chunk fits the chunk-size limit, no further splitting is done.

Here’s what the output could look like:

[![](https://substackcdn.com/image/fetch/$s_!5-DV!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb0e40cc1-996f-48f4-9306-781b112536e4_984x428.png)](https://substackcdn.com/image/fetch/$s_!5-DV!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb0e40cc1-996f-48f4-9306-781b112536e4_984x428.png)

As shown above:

- First, we define two chunks (the two paragraphs in purple).

- Next, paragraph 1 is further split into smaller chunks.


Unlike fixed-size chunks, this approach also maintains the natural flow of language and preserves complete ideas.

However, there is some extra overhead in terms of implementation and computational complexity.

* * *

### 4) Document structure-based chunking

This is another intuitive approach.

[![](https://substackcdn.com/image/fetch/$s_!NtgT!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe8febecd-ee68-42ff-ab06-41a0a3a43cd3_1102x306.gif)](https://substackcdn.com/image/fetch/$s_!NtgT!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe8febecd-ee68-42ff-ab06-41a0a3a43cd3_1102x306.gif)

It utilizes the inherent structure of documents, like headings, sections, or paragraphs, to define chunk boundaries.

This way, it maintains structural integrity by aligning with the document’s logical sections.

Here’s what the output could look like:

[![](https://substackcdn.com/image/fetch/$s_!9CjT!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F40bdaf3b-601d-4357-bc7f-89b47f812097_1025x663.png)](https://substackcdn.com/image/fetch/$s_!9CjT!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F40bdaf3b-601d-4357-bc7f-89b47f812097_1025x663.png)

That said, this approach assumes that the document has a clear structure, which may not be true.

Also, chunks may vary in length, possibly exceeding model token limits. You can try merging it with recursive splitting.

* * *

### 5) LLM-based chunking

[![](https://substackcdn.com/image/fetch/$s_!jVmL!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4d1b6d60-8956-4030-8525-d899ee61a9d5_1140x198.gif)](https://substackcdn.com/image/fetch/$s_!jVmL!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4d1b6d60-8956-4030-8525-d899ee61a9d5_1140x198.gif)

Since every approach has upsides and downsides, why not use the LLM to create chunks?

The LLM can be prompted to generate semantically isolated and meaningful chunks.

Quite evidently, this method will ensure high semantic accuracy since the LLM can understand context and meaning beyond simple heuristics (used in the above four approaches).

The only problem is that it is the most computationally demanding chunking technique of all five techniques discussed here.

Also, since LLMs typically have a limited context window, that is something to be taken care of.

* * *

Each technique has its own advantages and trade-offs.

I have observed that semantic chunking works pretty well in many cases, but again, you need to test.

The choice will heavily depend on the nature of your content, the capabilities of the embedding model, computational resources, etc.

We shall be doing a hands-on demo of these strategies pretty soon.

In the meantime, in case you missed it, yesterday, we discussed techniques to build robust NLP systems that rely on pairwise content similarity (RAG is one of them).

Read here: **[Bi-encoders and Cross-encoders for Sentence Pair Similarity Scoring – Part 1](https://www.dailydoseofds.com/bi-encoders-and-cross-encoders-for-sentence-pair-similarity-scoring-part-1/)**.

👉 Over to you: What other chunking strategies do you know?

Thanks for reading Daily Dose of Data Science! Subscribe below and receive a free data science PDF (530+ pages) with 150+ core data science and machine learning lessons.

Subscribe

* * *

### **P.S. For those wanting to develop “Industry ML” expertise:**

[![](https://substackcdn.com/image/fetch/$s_!cn8y!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F939bede7-b0de-4770-a3e9-34d39488e776_2733x1020.png)](https://substackcdn.com/image/fetch/$s_!cn8y!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F939bede7-b0de-4770-a3e9-34d39488e776_2733x1020.png)

At the end of the day, all businesses care about _impact_. That’s it!

- Can you reduce costs?

- Drive revenue?

- Can you scale ML models?

- Predict trends before they happen?


We have discussed several other topics (with implementations) in the past that align with such topics.

[Develop "Industry ML" Skills](https://www.dailydoseofds.com/membership)

Here are some of them:

- Learn sophisticated graph architectures and how to train them on graph data: [A Crash Course on Graph Neural Networks – Part 1](https://www.dailydoseofds.com/a-crash-course-on-graph-neural-networks-implementation-included/)

- Learn techniques to run large models on small devices: [Quantization: Optimize ML Models to Run Them on Tiny Hardware](https://www.dailydoseofds.com/quantization-optimize-ml-models-to-run-them-on-tiny-hardware/)

- Learn how to generate prediction intervals or sets with strong statistical guarantees for increasing trust: [Conformal Predictions: Build Confidence in Your ML Model’s Predictions](https://www.dailydoseofds.com/conformal-predictions-build-confidence-in-your-ml-models-predictions/).

- Learn how to identify causal relationships and answer business questions: [A Crash Course on Causality – Part 1](https://www.dailydoseofds.com/a-crash-course-on-causality-part-1/)

- Learn how to scale ML model training: [A Practical Guide to Scaling ML Model Training](https://www.dailydoseofds.com/how-to-scale-model-training/).

- Learn techniques to reliably roll out new models in production: [5 Must-Know Ways to Test ML Models in Production (Implementation Included)](https://www.dailydoseofds.com/5-must-know-ways-to-test-ml-models-in-production-implementation-included/)

- Learn how to build privacy-first ML systems: [Federated Learning: A Critical Step Towards Privacy-Preserving Machine Learning](https://www.dailydoseofds.com/federated-learning-a-critical-step-towards-privacy-preserving-machine-learning/).

- Learn how to compress ML models and reduce costs: [Model Compression: A Critical Step Towards Efficient Machine Learning](https://www.dailydoseofds.com/model-compression-a-critical-step-towards-efficient-machine-learning/).


All these resources will help you cultivate key skills that businesses and companies care about the most.

* * *

### **SPONSOR US**

Get your product in front of 100,000 data scientists and other tech professionals.

Our newsletter puts your products and services directly in front of an audience that matters — thousands of leaders, senior data scientists, machine learning engineers, data analysts, etc., who have influence over significant tech decisions and big purchases.

To ensure your product reaches this influential audience, reserve your space **[here](https://scorecard.dailydoseofds.com/sponsorship-assessment)** or reply to this email to ensure your product reaches this influential audience.

* * *

#### Subscribe to Daily Dose of Data Science

A free newsletter for continuous learning about data science and ML, lesser-known techniques, and how to apply them in 2 minutes. We keep things no-fluff.

Join 100,000+ data scientists from top companies like Google, NVIDIA, Microsoft, Uber, etc.

Subscribe

By subscribing, I agree to Substack's [Terms of Use](https://substack.com/tos), and acknowledge its [Information Collection Notice](https://substack.com/ccpa#personal-data-collected) and [Privacy Policy](https://substack.com/privacy).

[![Varun M's avatar](https://substackcdn.com/image/fetch/$s_!8MKR!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd7a062f9-2bf8-4413-a528-eb37a566d92d_144x144.png)](https://substack.com/profile/164101609-varun-m)

[![xyp's avatar](https://substackcdn.com/image/fetch/$s_!hLlZ!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe7d64984-8a3c-40e6-ad92-2e8b0468d10b_2082x2082.jpeg)](https://substack.com/profile/3529996-xyp)

[![Namrata Thakur's avatar](https://substackcdn.com/image/fetch/$s_!md1a!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3473fc16-60ab-481a-ade1-ffb4622bda65_144x144.png)](https://substack.com/profile/83626425-namrata-thakur)

[![Shihgian Lee's avatar](https://substackcdn.com/image/fetch/$s_!_skL!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0c35cd1e-80bf-4f5b-afa6-b2af4222faa8_144x144.png)](https://substack.com/profile/14383681-shihgian-lee)

[![Luis Moro's avatar](https://substackcdn.com/image/fetch/$s_!7_Ux!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffb0ef9d7-bc61-4b4f-9ed8-16b588225f44_96x96.png)](https://substack.com/profile/90164107-luis-moro)

130 Likes∙

[9 Restacks](https://substack.com/note/p-150416240/restacks?utm_source=substack&utm_content=facepile-restacks)

130

[5](https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag/comments)
9

Share

PreviousNext

#### Discussion about this post

CommentsRestacks

![User's avatar](https://substackcdn.com/image/fetch/$s_!TnFC!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack.com%2Fimg%2Favatars%2Fdefault-light.png)

[![alex's avatar](https://substackcdn.com/image/fetch/$s_!Gasv!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9e89c499-06e0-4961-9137-bd29ae10bc44_144x144.png)](https://substack.com/profile/280936789-alex?utm_source=comment)

[alex](https://substack.com/profile/280936789-alex?utm_source=substack-feed-item)

[Oct 27, 2024](https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag/comment/74310617 "Oct 27, 2024, 11:22 AM")

Thanks so much for sharing, BTW, I'd like to know what tool you used to draw the sketch in gif format, I really like the dynamics!

Expand full comment

Like (8)

Reply

Share

[1 reply](https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag/comment/74310617)

[![Abdul wasea's avatar](https://substackcdn.com/image/fetch/$s_!Re97!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcb1d7469-afac-41ce-8249-c200b3c547ad_144x144.png)](https://substack.com/profile/271315401-abdul-wasea?utm_source=comment)

[Abdul wasea](https://substack.com/profile/271315401-abdul-wasea?utm_source=substack-feed-item)

[Nov 9, 2024](https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag/comment/76339838 "Nov 9, 2024, 8:11 AM")

How to make gif Like this. Can you teach me How to create. Thanks for this post.

Expand full comment

Like (2)

Reply

Share

[3 more comments...](https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag/comments)

TopLatestDiscussions

[FREE Daily Dose of Data Science PDF](https://blog.dailydoseofds.com/p/free-daily-dose-of-data-science-pdf)

[Collection of posts on core DS/ML topics.](https://blog.dailydoseofds.com/p/free-daily-dose-of-data-science-pdf)

Apr 20, 2023•
[Avi Chawla](https://substack.com/@avichawla)

577

[22](https://blog.dailydoseofds.com/p/free-daily-dose-of-data-science-pdf/comments)

![](https://substackcdn.com/image/fetch/$s_!cWR1!,w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F13d99ac7-b629-4494-b7e9-8d7f3f969097_2820x4500.jpeg)

[9 MCP Projects for AI Engineers](https://blog.dailydoseofds.com/p/9-mcp-projects-for-ai-engineers)

[(explained with visuals & open-source code)](https://blog.dailydoseofds.com/p/9-mcp-projects-for-ai-engineers)

Jun 10•
[Avi Chawla](https://substack.com/@avichawla)

397

[6](https://blog.dailydoseofds.com/p/9-mcp-projects-for-ai-engineers/comments)

![](https://substackcdn.com/image/fetch/$s_!PJHI!,w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F39330805-8f88-421f-84f0-bf2029caf587_2307x2199.png)

[15 DS/ML Cheat Sheets](https://blog.dailydoseofds.com/p/15-dsml-cheat-sheets)

[Single frame summaries of must-know DS/ML concepts and techniques.](https://blog.dailydoseofds.com/p/15-dsml-cheat-sheets)

Sep 22, 2024•
[Avi Chawla](https://substack.com/@avichawla)

124

[View comments (0)](https://blog.dailydoseofds.com/p/15-dsml-cheat-sheets/comments)

![](https://substackcdn.com/image/fetch/$s_!vn65!,w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F86baf45d-d494-4af0-bf64-884f7858bf6d_1456x1749.png)

See all

Ready for more?

Subscribe