# Article Search

A search engine over blog and news feeds — RSS/Atom ingestion, BM25F ranking with
recency decay, a Flask API, and an evaluation harness that scores the ranking
instead of asserting it is good.

```bash
pip install -r requirements.txt
python cli.py search "bm25 ranking"   # search the sample corpus
python cli.py evaluate                # nDCG@10, precision, recall, MRR
python cli.py ablate                  # what each ranking component is worth
python bench.py --scale               # latency from 1k to 100k documents
python app.py                         # http://localhost:5000
pytest                                # 68 tests
```

No external services. The default index is in-process, so a fresh clone runs
every command above — including the full test suite and the evaluation — with
nothing installed but Flask and pytest.

---

## The decision this is built around

Most search repositories require a running Elasticsearch cluster, which means
nobody clones them and nobody tests the ranking. That is a real cost: ranking
is the part worth reviewing, and it is the part an unrunnable repository hides.

So the index sits behind a protocol with two implementations:

| Backend | When | Tested in CI |
|---|---|---|
| `BM25Index` | default; in-process, zero dependencies | yes |
| `ElasticIndex` | `--backend elasticsearch`, real cluster | no — needs a cluster |

Ranking, evaluation and the API are written once against `SearchIndex` rather
than twice against two engines. The trade-off is that the BM25 implementation
is mine to maintain and mine to get wrong — which is why the scoring has tests
that assert specific numerical properties rather than just "returns results".

## Ranking

BM25 per field, combined with weights, then multiplied by a recency factor.

```
relevance = w_title · BM25(title) + w_body · BM25(body)
recency   = (1 − floor) + floor · 2^(−age_days / half_life)
final     = relevance · recency
```

**Why per-field rather than one concatenated field.** BM25's length
normalisation would treat a title as a handful of tokens inside a long
document and effectively erase it. A term in the title is strong evidence
about what a post is *about*; the same term once in a 2,000-word body is
usually incidental.

**Why the recency factor is multiplicative and floored.** An additive
recency bonus knows nothing about the query, so it can lift an irrelevant
recent post above a relevant older one. A multiplier can only reorder
documents that already matched, which keeps relevance dominant. The floor
(default 0.5) stops decay from tending to zero — blog archives are full of
posts that are both ancient and definitive, and an unfloored decay buries
them permanently.

**Why undated documents are neutral rather than old.** A missing `pubDate`
is a property of the feed, not of the article. Treating it as maximally old
would systematically penalise every feed that omits dates.

## Ingestion

Three properties, each of which cost the naive version something:

- **Per-feed failure isolation.** Feeds break constantly — dead DNS, expired
  certificates, XML that doesn't parse. `fetch_all` records the failure and
  continues; the caller gets both the documents and the failures. A crawl that
  raises on the first bad feed indexes nothing.
- **Conditional GET.** ETag and Last-Modified from the previous fetch are sent
  back, so an unchanged feed answers 304 with no body.
- **Dedup by content, not URL.** The document id is a hash of normalised
  title and body. The same post syndicated to a canonical site, a Medium
  mirror and an aggregator collapses to one document; a URL-keyed index stores
  three.

## Evaluation — and what it actually shows

`eval/queries.json` holds 24 queries over the 165-document sample corpus, with
graded relevance labels: 2 = directly on topic, 1 = related, 0 = irrelevant.

```
nDCG@10       0.934
precision@10  0.946
recall@10     0.191
MRR           1.000
```

**That nDCG is not evidence of a good ranker, and this is the interesting part
of the repository.** Run `python cli.py ablate`:

| configuration | nDCG@10 | P@10 | MRR | |
|---|---|---|---|---|
| full | 0.934 | 0.946 | 1.000 | BM25F (title ×2.5) + recency decay |
| no title boost | **0.950** | 0.958 | 1.000 | title and body weighted equally |
| no recency decay | 0.942 | 0.950 | 1.000 | relevance only |
| shuffled baseline | 0.887 | 0.904 | 0.972 | same candidates, random order |

Three things fall out of that table, none of them flattering:

1. **The shuffled baseline scores 0.887.** It retrieves the same candidates and
   destroys only the ordering. A gap of 0.047 between "correctly ranked" and
   "randomly ranked" means this evaluation is mostly measuring the retrieval
   filter, not the ranking — so 0.934 is close to what the corpus hands you for
   free.
2. **The title boost is not supported.** Weighting title and body equally
   scores *higher*. A sweep shows the metric is flat once the weight exceeds 1:

   ```
   1.0  0.950     2.5  0.934
   1.5  0.934     3.0  0.934
   2.0  0.934     4.0  0.934
   ```
3. **Recency decay is worth 0.008**, which on 24 queries is noise.

I kept `title_weight = 2.5` anyway, and that is a judgement call rather than an
oversight. The sample corpus has formulaic titles generated from five templates,
which makes title matching unrepresentative of real feed data, where titles are
the strongest signal available. The honest summary is not "the title boost is
wrong" — it is **this eval set cannot tell**, and a flat sweep is what "cannot
tell" looks like. Fixing that needs labels over a real crawl, which is the first
item under Limitations.

Recall@10 is 0.191 because each query has ~52 documents labelled relevant
(~21 on-topic at grade 2, the rest neighbouring topics at grade 1) and k is 10.
The ceiling at k=10 is therefore 0.194, and the system is essentially at it. Recall is
reported for completeness, not as a target — a system returning 52 results per
query would score better on it and be worse to use.

## Performance

`python bench.py --scale`, in-process BM25, Apple Silicon, single process:

| documents | build | p50 | p95 | p99 |
|---|---|---|---|---|
| 1,000 | 7 ms | 0.10 ms | 0.20 ms | 0.22 ms |
| 10,000 | 78 ms | 0.72 ms | 2.01 ms | 2.20 ms |
| 50,000 | 400 ms | 4.15 ms | 12.14 ms | 12.58 ms |
| 100,000 | 861 ms | 8.67 ms | 26.35 ms | 28.81 ms |

Scaling is linear in corpus size, which is expected: scoring walks the postings
list of every query term. p50/p95/p99 rather than a mean, because a mean hides
the tail and the tail is what a user notices.

The larger corpora are the sample corpus replicated with per-copy token
variation so that content hashes differ — otherwise dedup would collapse them
back and the benchmark would measure nothing.

## Limitations

- **The shipped corpus is synthetic.** Real blog posts cannot be redistributed,
  and a corpus that changes whenever a feed updates makes nDCG incomparable
  between runs. `tools/build_sample_corpus.py` generates it deterministically;
  `python cli.py crawl data/feeds.json` builds a real one. Every number above
  therefore describes the ranker *on an easy corpus*, and the ablation table is
  the honest statement of how much that is worth.
- **Lexical matching only.** No stemming and no embeddings, so "profiling" does
  not match "profiler" and a paraphrased query misses entirely. The index
  protocol is the seam where a dense retriever would go.
- **The index is in memory and rebuilt at startup.** Fine to roughly 100k
  documents on one process, per the table above. Beyond that, or with more than
  one writer, the Elasticsearch backend is the reason it exists.
- **No incremental crawl state on disk.** `FeedState` holds ETags in memory for
  the life of a process, so a fresh run re-fetches everything.

## Layout

```
src/
  documents.py     normalisation, tokenisation, content-hash identity
  feeds.py         RSS/Atom parsing, conditional GET, per-feed isolation
  index/
    bm25.py        in-process BM25F
    elastic.py     Elasticsearch backend, same protocol
  rank.py          recency decay and final ordering
  search.py        query pipeline (overfetch, then re-rank)
  evaluate.py      nDCG@k, precision, recall, MRR
  ablation.py      component ablations and the title-weight sweep
app.py             Flask API and single-page UI
cli.py             search / evaluate / ablate / crawl
bench.py           latency, including --scale
tools/             deterministic sample-corpus generator
```

## Stack

Python 3.11+ · Flask · pytest. Elasticsearch optional.

## License

MIT
