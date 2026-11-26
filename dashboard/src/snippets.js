// Examples that a person can copy. The key is always a placeholder: a real key is never put in an example.
export function powershellSnippet(gatewayUrl, keyPlaceholder = "ap-YOUR-KEY") {
  return [
    `$key = "${keyPlaceholder}"`,
    `$body = @{ model = "auto"; messages = @(@{ role = "user"; content = "What is the capital of France?" }) } | ConvertTo-Json -Depth 5`,
    `Invoke-RestMethod -Method Post -Uri ${gatewayUrl}/v1/chat/completions -ContentType "application/json" -Headers @{ Authorization = "Bearer $key" } -Body $body`,
  ].join("\n");
}

export function curlSnippet(gatewayUrl, keyPlaceholder = "ap-YOUR-KEY") {
  return [
    `curl ${gatewayUrl}/v1/chat/completions \\`,
    `  -H "Authorization: Bearer ${keyPlaceholder}" \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'`,
  ].join("\n");
}