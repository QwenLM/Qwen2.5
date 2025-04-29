# Ollama

:::{attention}
To be updated for Qwen3.
:::

[Ollama](https://ollama.com/) helps you run LLMs locally with only a few commands.
It is available at macOS, Linux, and Windows.
Now, Qwen2.5 is officially on Ollama, and you can run it with one command:

```bash
ollama run qwen2.5
```

Next, we introduce more detailed usages of Ollama for running Qwen2.5 models.

## Quickstart

Visit the official website [Ollama](https://ollama.com/) and click download to install Ollama on your device.
You can also search models on the website, where you can find the Qwen2.5 models.
Except for the default one, you can choose to run Qwen2.5-Instruct models of different sizes by:

- `ollama run qwen2.5:0.5b`
- `ollama run qwen2.5:1.5b`
- `ollama run qwen2.5:3b`
- `ollama run qwen2.5:7b`
- `ollama run qwen2.5:14b`
- `ollama run qwen2.5:32b`
- `ollama run qwen2.5:72b`

:::{note}
`ollama` does not host base models.
Even though the tag may not have the instruct suffix, they are all instruct models.
:::

## Run Ollama with Your GGUF Files

Sometimes you don't want to pull models and you just want to use Ollama with your own GGUF files.
Suppose you have a GGUF file of Qwen2.5, `qwen2.5-7b-instruct-q5_0.gguf`.
For the first step, you need to create a file called `Modelfile`.
The content of the file is shown below:

```text
FROM qwen2.5-7b-instruct-q5_0.gguf

# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER repeat_penalty 1.05
PARAMETER top_k 20

TEMPLATE """{{ if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{ .System }}
{{- if .Tools }}

# Tools

You are provided with function signatures within <tools></tools> XML tags:
<tools>{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""

# set the system message
SYSTEM """You are Qwen, created by Alibaba Cloud. You are a helpful assistant."""
```

Then create the Ollama model by running:

```bash
ollama create qwen2.5_7b -f Modelfile
```

Once it is finished, you can run your Ollama model by:

```bash
ollama run qwen2.5_7b
```

## Tool Use

Tool use is now supported Ollama and you should be able to run Qwen2.5 models with it.
For more details, see our [function calling guide](../framework/function_call).