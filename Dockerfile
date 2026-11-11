# Multi-stage Dockerfile for AgentoCode (Node 22 Debian glibc for onnxruntime-node compatibility)

FROM node:22-bookworm-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM deps AS test
COPY tsconfig.json esbuild.mjs ./
COPY package.json package-lock.json .gitignore .vscodeignore .dockerignore docker-compose.yml Dockerfile README.md .env.example ./
COPY .github ./.github
COPY media ./media
COPY src ./src
COPY test ./test
RUN npm run typecheck && npm test && npm run build && node --test dist-test/*.test.cjs
CMD ["node", "--test", "dist-test/*.test.cjs"]

FROM node:22-bookworm-slim AS prod-deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force

FROM node:22-bookworm-slim AS worker
WORKDIR /app

RUN mkdir -p /workspace /state /hf-cache && chown -R node:node /workspace /state /hf-cache

COPY --from=prod-deps /app/package.json ./package.json
COPY --from=prod-deps /app/node_modules ./node_modules
COPY --from=test /app/dist/indexerWorker.cjs ./dist/indexerWorker.cjs

ENV NODE_ENV=production \
    AGENTOCODE_WORKSPACE_ROOT=/workspace \
    AGENTOCODE_STATE_DIR=/state \
    AGENTOCODE_KNOWLEDGE_DIR=/state/knowledge_base \
    AGENTOCODE_MODEL_CACHE_DIR=/hf-cache \
    AGENTOCODE_EMBED_PROVIDER=transformers_onnx \
    AGENTOCODE_EMBED_MODEL_ID=nomic-ai/nomic-embed-text-v1.5 \
    AGENTOCODE_EMBED_MODEL_REVISION=e9b6763023c676ca8431644204f50c2b100d9aab \
    AGENTOCODE_EMBED_DTYPE=q8 \
    GUARDIAN_LOCAL_FILES_ONLY=0

USER node
VOLUME ["/workspace", "/state", "/hf-cache"]

ENTRYPOINT ["node", "/app/dist/indexerWorker.cjs"]

