# Сборка статики Vite и раздача через nginx (см. корневой docker-compose.yml).
FROM node:20-alpine AS build
WORKDIR /app

COPY DC_digital_twin/web/package.json DC_digital_twin/web/package-lock.json ./
RUN npm ci

COPY DC_digital_twin/web ./

# URL с точки зрения браузера пользователя (хост): порты как в compose.
ARG VITE_API_BASE=http://127.0.0.1:8000
ARG VITE_ORCHESTRATOR_BASE=http://127.0.0.1:8030
ENV VITE_API_BASE=$VITE_API_BASE
ENV VITE_ORCHESTRATOR_BASE=$VITE_ORCHESTRATOR_BASE

RUN npm run build

FROM nginx:1.27-alpine
COPY DC_digital_twin/web/nginx.default.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
