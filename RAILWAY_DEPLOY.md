# Railway Deployment Guide

## Prerequisites
- Railway account (sign up at https://railway.app/)
- GitHub repository: https://github.com/AbhishekItmc/dermatology-poc

## Deployment Steps

### 1. Create New Project
1. Go to https://railway.app/
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Connect your GitHub account if not already connected
5. Select repository: `AbhishekItmc/dermatology-poc`

### 2. Add PostgreSQL Database
1. In your project, click **"New"** → **"Database"** → **"Add PostgreSQL"**
2. Railway will automatically create a PostgreSQL instance
3. Note: Connection details are automatically available as environment variables

### 3. Add Redis
1. Click **"New"** → **"Database"** → **"Add Redis"**
2. Railway will automatically create a Redis instance
3. Connection details are automatically available

### 4. Deploy Backend
1. Click **"New"** → **"GitHub Repo"** → Select your repo
2. Set **Root Directory**: `backend`
3. Add environment variables:
   - `ENVIRONMENT` = `production`
   - `SECRET_KEY` = (generate random 32-char string)
   - `ENCRYPTION_KEY` = (generate random 32-char string)
   - `POSTGRES_SERVER` = `${{Postgres.PGHOST}}`
   - `POSTGRES_USER` = `${{Postgres.PGUSER}}`
   - `POSTGRES_PASSWORD` = `${{Postgres.PGPASSWORD}}`
   - `POSTGRES_DB` = `${{Postgres.PGDATABASE}}`
   - `DATABASE_URL` = `${{Postgres.DATABASE_URL}}`
   - `REDIS_HOST` = `${{Redis.REDIS_HOST}}`
   - `REDIS_PORT` = `${{Redis.REDIS_PORT}}`
4. Railway will auto-detect Dockerfile and deploy

### 5. Deploy Frontend
1. Click **"New"** → **"GitHub Repo"** → Select your repo again
2. Set **Root Directory**: `frontend`
3. Add environment variable:
   - `REACT_APP_API_URL` = `https://[your-backend-url].railway.app/api/v1`
4. Railway will auto-detect Dockerfile and deploy

### 6. Configure Networking
1. For backend service:
   - Click on the service → **Settings** → **Networking**
   - Click **"Generate Domain"** to get public URL
2. For frontend service:
   - Click on the service → **Settings** → **Networking**
   - Click **"Generate Domain"** to get public URL
3. Update frontend's `REACT_APP_API_URL` with the backend URL

### 7. Test Deployment
1. Visit your frontend URL
2. Login with:
   - Username: `admin`
   - Password: `admin123`

## Railway Advantages
- **Simpler setup**: No Blueprint configuration needed
- **Better free tier**: $5/month credit (enough for small apps)
- **Auto-detection**: Automatically detects Dockerfiles
- **Easy scaling**: Simple UI for scaling services
- **Built-in monitoring**: Logs and metrics included
- **No Redis limit**: Can create multiple Redis instances

## Troubleshooting

### Build Failures
- Check build logs in Railway dashboard
- Ensure Dockerfiles are in correct directories
- Verify all dependencies are in requirements.txt

### Connection Issues
- Verify environment variables are set correctly
- Check that services are in the same project (for internal networking)
- Ensure frontend has correct backend URL

### Database Issues
- Railway automatically runs migrations on deploy
- Check PostgreSQL logs if connection fails
- Verify DATABASE_URL format is correct

## Cost Estimate (Free Tier)
- PostgreSQL: ~$1/month
- Redis: ~$0.50/month
- Backend: ~$2/month
- Frontend: ~$1.50/month
- **Total**: ~$5/month (covered by free credit)

## Support
- Railway Docs: https://docs.railway.app/
- Railway Discord: https://discord.gg/railway
