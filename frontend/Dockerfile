# ── Build stage ────────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app

COPY package*.json ./
RUN npm ci --frozen-lockfile

COPY . .
RUN npm run build

# ── Serve with Nginx ───────────────────────────────────────────────────────────
FROM nginx:1.26-alpine AS runtime

COPY --from=builder /app/dist /usr/share/nginx/html

# SPA routing: all unknown paths → index.html
COPY docker/nginx-spa.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
