# gpgpujson

General purpose GPU JSON parsing

This project parses JSON entirely on the GPU via wgpu compute shaders. The parser decomposes JSON parsing into a pipeline of parallel prefix scans, producing a flat tape of structual characters

This is purely for research into reducing JSON parsing into what Raph Levein calls invitingly parallel problems

# Usage

```sh
cargo r --release testdata/hits_sample.json
```

# Reading

https://raphlinus.github.io/gpu/2020/09/05/stack-monoid.html<br>
https://raphlinus.github.io/personal/2018/05/10/toward-gpu-json-parsing.html<br>
