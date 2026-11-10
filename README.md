# labelswap

Stop paying a frontier model to pick one label out of twenty. labelswap learns from the labels your LLM has already produced, handles the easy calls locally in microseconds, and defers to the LLM only when it isn't confident.

It is one small Python package with no dependencies and no API key.

## Quick start

```
pip install git+https://github.com/perron/labelswap
python examples/make_tickets.py            # writes a made-up demo log
labelswap train examples/tickets.jsonl --target 0.98 --cost 0.02
labelswap decide labelswap.json --text "checkout is down and returns 503 errors"
```

Input is JSONL with one logged LLM call per line: `{"text": "...", "label": "..."}`. Use `--text-field` and `--label-field` if your log uses other names.

## How it works

1. `train` fits a multinomial Naive Bayes model (words plus word pairs) on most of your log and holds back a deterministic slice.
2. On that slice it picks the lowest confidence gate where the local answers still agree with the LLM at your `--target` rate. If no gate reaches the target, every call defers.
3. `decide` returns `{"label", "confidence", "action"}`, where `action` is `local` or `defer`. Call your LLM only on `defer`.
4. `shadow` runs the model next to fresh LLM labels, so you can watch agreement before you trust it. The tuning numbers from `train` are measured on the same slice that set the gate, so treat them as optimistic until `shadow` confirms them on new data.

Demo output on the synthetic ticket log (600 made-up rows, 5 labels):

```
handled locally:       136 (96.5%)
agreement when local:  98.5%
target agreement:      98.0% (met)
```

That demo data is deliberately easy. Real logs will cover less, and that's the point of the gate.

## Rollout pattern

Run it in shadow first: keep calling the LLM, log both answers, and run `labelswap shadow` weekly. Switch to `decide` once agreement holds, and keep deferring the unsure calls.

## Credit

Inspired by the Strands Agents post [Amazon saves millions by replacing Sonnet 4.6 with Strands Decider](https://strandsagents.com/blog/amazon-saves-millions-replacing-sonnet-with-strands-decider/), which described swapping an LLM ticket labeler for a decision model, running both side by side first, and tuning a confidence threshold. labelswap is an original, clean-room tool. It shares no code with Strands, uses a far simpler model, and is not affiliated with Amazon or AWS.

## Point your agents at it

Agents can read [AGENTS.md](AGENTS.md) and wire `labelswap decide` in front of any classification prompt. Or write your own version; the whole model is about 150 lines.

## Contributing

Issues and pull requests are welcome. Run `python -m unittest discover -s tests` before sending one, and keep the package dependency-free.

## License

MIT
