FROM ghcr.io/eosc-data-commons/eoscdcpoc-frontend:latest AS frontend

FROM rust:trixie AS build

RUN apt-get update && apt-get -y install libssl-dev curl build-essential

COPY . /app

# Copy frontend files from the frontend image
COPY --from=frontend /webapp /app/src/webapp

WORKDIR /app


# Might need some flags, see https://crates.io/crates/ort
# ENV RUSTFLAGS="-Clink-args=-Wl,-rpath,\$ORIGIN"

RUN --mount=type=cache,target=/root/.cargo cargo build --release


FROM debian:trixie AS runtime

# Install libssl3 for libssl.so.3 runtime dependency and CA certificates to ddl ONNX model files
RUN apt-get update && apt-get -y install libssl3 ca-certificates

COPY --from=build /app/src/webapp /src/webapp
COPY --from=build /app/target/release/data-commons-mcp /
EXPOSE 8000
CMD ["./data-commons-mcp"]
