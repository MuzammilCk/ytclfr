# Stage 1: Build the React Application
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies based on package.json and package-lock.json
COPY package*.json ./
RUN npm install

# Copy application source code
COPY . .

# Build for production
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine

# Copy the build output to Nginx's static file serving directory
COPY --from=builder /app/dist /usr/share/nginx/html

# Overwrite default Nginx config to support client-side routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
