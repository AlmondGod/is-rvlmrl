"""Bounded-memory reading of the recorded top-level JSON trace array."""
import json

def iter_trace(path):
 decoder=json.JSONDecoder();buffer='';started=False;eof=False
 with open(path) as stream:
  while True:
   if not buffer.strip() and not eof:
    chunk=stream.read(1024*1024);eof=not chunk;buffer+=chunk
   buffer=buffer.lstrip()
   if not started:
    if not buffer.startswith('['):raise ValueError('Expected trace array')
    buffer=buffer[1:];started=True;continue
   buffer=buffer.lstrip(' \r\n\t,')
   if buffer.startswith(']'):return
   try:value,end=decoder.raw_decode(buffer)
   except json.JSONDecodeError:
    if eof:raise
    chunk=stream.read(1024*1024);eof=not chunk;buffer+=chunk;continue
   yield value;buffer=buffer[end:]
