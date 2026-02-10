# Deploy to Render.com

## Quick Deployment Steps

### 1. Prerequisites
- GitHub account
- Render.com account (free - sign up at https://render.com)

### 2. Push Code to GitHub

If you haven't already, push your code to GitHub:

```powershell
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Dermatological Analysis PoC"

# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### 3. Deploy on Render

#### Option A: Using Blueprint (Recommended - Deploys Everything)

1. Go to https://render.com/dashboard
2. Click "New" → "Blueprint"
3. Connect your GitHub repository
4. Render will automatically detect `render.yaml` and create all services
5. Click "Apply" to deploy

#### Option B: Manual Deployment (Deploy Services Individually)

**Step 1: Deploy PostgreSQL**
1. Go to https://render.com/dashboard
2. Click "New" → "PostgreSQL"
3. Name: `dermatology-postgres`
4. Database: `dermatology_poc`
5. User: `postgres`
6. Plan: Free
7. Click "Create Database"

**Step 2: Deploy Redis**
1. Click "New" → "Redis"
2. Name: `dermatology-redis`
3. Plan: Free
4. Click "Create Redis"

**Step 3: Deploy Backend**
1. Click "New" → "Web Service"
2. Connect your GitHub repository
3. Name: `dermatology-backend`
4. Root Directory: `backend`
5. Environment: Docker
6. Dockerfile Path: `backend/Dockerfile`
7. Plan: Free
8. Add Environment Variables:
   - `POSTGRES_SERVER`: (copy from PostgreSQL internal connection string)
   - `POSTGRES_USER`: `postgres`
   - `POSTGRES_PASSWORD`: (copy from PostgreSQL)
   - `POSTGRES_DB`: `dermatology_poc`
   - `REDIS_HOST`: (copy from Redis internal connection string)
   - `REDIS_PORT`: `6379`
   - `ENVIRONMENT`: `production`
   - `SECRET_KEY`: (generate random 32-char string)
   - `ENCRYPTION_KEY`: (generate random 32-char string)
9. Click "Create Web Service"

**Step 4: Deploy Frontend**
1. Click "New" → "Web Service"
2. Connect your GitHub repository
3. Name: `dermatology-frontend`
4. Root Directory: `frontend`
5. Environment: Docker
6. Dockerfile Path: `frontend/Dockerfile`
7. Plan: Free
8. Add Environment Variable:
   - `REACT_APP_API_URL`: (copy backend URL, e.g., `https://dermatology-backend.onrender.com/api/v1`)
9. Click "Create Web Service"

### 4. Access Your Application

After deployment completes (5-10 minutes):

- **Frontend**: `https://dermatology-frontend.onrender.com`
- **Backend API**: `https://dermatology-backend.onrender.com`
- **API Docs**: `https://dermatology-backend.onrender.com/api/v1/docs`

### 5. Test Login

Use these credentials:
- Username: `admin`
- Password: `admin123`

## Important Notes

### Free Tier Limitations
- Services spin down after 15 minutes of inactivity
- First request after spin-down takes 30-60 seconds to wake up
- 750 hours/month free (enough for demo purposes)

### Troubleshooting

**If backend fails to start:**
1. Check logs in Render dashboard
2. Verify all environment variables are set correctly
3. Ensure PostgreSQL and Redis are running before backend

**If frontend can't connect to backend:**
1. Check `REACT_APP_API_URL` environment variable
2. Ensure it points to the correct backend URL
3. Rebuild frontend after changing environment variables

### Upgrading to Paid Plan

For production use:
- Upgrade to Starter plan ($7/month per service)
- Services stay always-on
- Better performance and reliability

## Alternative: Deploy Backend Only

If you only want to deploy the backend API:

1. Deploy PostgreSQL (Step 1 above)
2. Deploy Redis (Step 2 above)
3. Deploy Backend (Step 3 above)
4. Access API at: `https://dermatology-backend.onrender.com/api/v1/docs`

You can then run the frontend locally and point it to the deployed backend.

## Support

For issues with Render deployment:
- Render Docs: https://render.com/docs
- Render Community: https://community.render.com
