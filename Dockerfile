FROM rust:alpine AS build

# Install necessary build dependencies for Alpine
RUN apk add --no-cache \
    musl-dev \
    gcc \
    libc-dev \
    pkgconfig \
    curl \
    openssl-dev \
    openssl-libs-static

COPY . /app

WORKDIR /app

RUN cargo build --release


FROM alpine AS runtime

COPY --from=build /app/target/release/data-commons-mcp /
EXPOSE 3000
CMD ["./data-commons-mcp"]
