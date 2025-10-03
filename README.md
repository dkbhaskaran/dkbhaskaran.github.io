# Dinesh's blog using Chirpy Starter

[![Gem Version](https://img.shields.io/gem/v/jekyll-theme-chirpy)][gem]&nbsp;
![GitHub license](https://img.shields.io/github/license/cotes2020/chirpy-starter.svg?color=blue)

My attempt to blogging.

## Usage

Check out the [blog](https://dkbhaskaran.github.io/)

### 🛠️ Testing

For testing on a local machine, follow the below steps

#### 📦 Using the latest Ubuntu-based container, install the necessary dependencies

```bash
$ git clone https://github.com/dkbhaskaran/dkbhaskaran.github.io && cd dkbhaskaran.github.io
$ docker run --rm -it -v "$PWD:/srv/jekyll" -p 4000:4000 ubuntu:latest
$ apt-get update
$ DEBIAN_FRONTEND=noninteractive apt-get install -y Node.js npm ruby-dev jekyll
```

#### 🚀 Run the Jekyll server

```bash
$ cd /srv/jekyll
# if git complains about dubious ownership
$ git config --global --add safe.directory /srv/jekyll/dkbhaskaran.github.io
$ bundle install
$ bundle exec htmlproofer _site
$ bundle exec jekyll serve --host 0.0.0.0
```

## License

This work is published under [MIT License](!License).

[gem]: https://rubygems.org/gems/jekyll-theme-chirpy
[chirpy]: https://github.com/cotes2020/jekyll-theme-chirpy/
