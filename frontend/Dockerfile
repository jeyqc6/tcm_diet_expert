# diet_expert · 前端最小闭环镜像

FROM node:22-slim AS build
WORKDIR /app
# Baked into client JS at build time. Default matches local docker compose;
# VPS deploy sets PUBLIC_API_BASE_URL in root .env → compose build-arg.
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8123
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:22-slim
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/public ./public
COPY --from=build /app/.next ./.next
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package.json ./package.json

EXPOSE 3000
CMD ["npm", "run", "start"]
