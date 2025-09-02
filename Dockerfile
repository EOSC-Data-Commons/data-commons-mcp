FROM rust:trixie AS build

RUN apt-get update && apt-get -y install libssl-dev curl build-essential

COPY . /app

WORKDIR /app


# Might need some flags, see https://crates.io/crates/ort
# ENV RUSTFLAGS="-Clink-args=-Wl,-rpath,\$ORIGIN"

# RUN cargo build --release
RUN --mount=type=cache,target=/root/.cargo cargo build --release


FROM debian:trixie AS runtime

# Install libssl3 for libssl.so.3 runtime dependency and CA certificates to ddl ONNX model files
RUN apt-get update && apt-get -y install libssl3 ca-certificates

COPY --from=build /app/target/release/data-commons-mcp /
EXPOSE 8000
CMD ["./data-commons-mcp"]
