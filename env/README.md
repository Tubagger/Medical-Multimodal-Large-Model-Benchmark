# API Credentials

Copy `apikey.example.yaml` to `apikey.yaml`:

```bash
cp env/apikey.example.yaml env/apikey.yaml
```

The code loads the YAML mapping into environment variables. The names currently
used by the model adapters are `openai_apikey`, `anthropic_apikey`,
`google_apikey`, `qwen_apikey`, `deepseek_apikey`, and `ernie_apikey`.

`env/apikey.yaml` is ignored by Git. Do not place production credentials in the
repository, issue attachments, notebooks, or logs.
