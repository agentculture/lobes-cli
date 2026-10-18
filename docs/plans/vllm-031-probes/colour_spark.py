import base64,json,struct,urllib.request,zlib,subprocess,re,pathlib
key=""
for l in pathlib.Path.home().joinpath(".lobes/.env").read_text().splitlines():
    m=re.match(r"^GATEWAY_API_KEY=(.*)$",l)
    if m: key=m.group(1).strip()
def png(rgb,n):
    raw=b"".join(b"\x00"+bytes(rgb)*n for _ in range(n))
    c=lambda t,d: struct.pack(">I",len(d))+t+d+struct.pack(">I",zlib.crc32(t+d)&0xffffffff)
    return b"\x89PNG\r\n\x1a\n"+c(b"IHDR",struct.pack(">IIBBBBB",n,n,8,2,0,0,0))+c(b"IDAT",zlib.compress(raw))+c(b"IEND",b"")
ip=subprocess.run(["docker","inspect","-f","{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}","model-gear-vllm-primary"],capture_output=True,text=True).stdout.strip()
ok=0;n_all=0
for url,model,auth in (("http://localhost:8001","cortex-spark",True),("http://%s:8000"%ip,"unsloth/Qwen3.8-27B-NVFP4",False)):
  for n in (64,224):
    for name,rgb in (("red",(255,0,0)),("blue",(0,0,255)),("green",(0,160,0))):
      b={"model":model,"max_tokens":300,"temperature":0,"chat_template_kwargs":{"enable_thinking":False},"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(png(rgb,n)).decode()}},{"type":"text","text":"What colour is this image? Answer with one word."}]}]}
      h={"Content-Type":"application/json"}
      if auth and key: h["Authorization"]="Bearer "+key
      r=json.load(urllib.request.urlopen(urllib.request.Request(url+"/v1/chat/completions",json.dumps(b).encode(),h),timeout=300))
      a=r["choices"][0]["message"]["content"].strip(); p=name in a.lower(); ok+=p; n_all+=1
      print(("gateway cortex-spark" if auth else "lane (direct)       "),n,"px",name,"->",a[:30],"PASS" if p else "FAIL")
print(f"{ok}/{n_all}")
