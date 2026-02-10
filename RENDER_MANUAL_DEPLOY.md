# Manual Render Deployment Guide

Since the Blueprint is having issues, let's deploy each service manually. This is actually more reliable!

## Step 1: Deploy PostgreSQL Database

1. Go to https://render.com/dashboard
2. Click **"New"** → **"PostgreSQL"**
3. Fill in:
   - **Name**: `dermatology-postgres`
   - **Database**: `dermatology_poc`
   - **Region**: Choose closest to you
   - **Plan**: **Free**
4. Click **"Create Database"**
5. Wait 2-3 minutes for it to be ready
6. **IMPORTANT**: Copy the **Internal Database URL** (you'll need this later)

## Step 2: Deploy Redis

1. Click **"New"** → **"Redis"**
2. Fill in:
   - **Name**: `dermatology-redis`
   - **Region**: Same as database
   - **Plan**: **Free**
   - **Max Memory Policy**: `noeviction`
3. Click **"Create Redis"**
4. Wait 1-2 minutes
5. **IMPORTANT**: Copy the **Internal Redis URL** (you'll need this later)

## Step 3: Deploy Backend API

1. Click **"New"** → **"Web Service"**
2. Connect your GitHub account if not already connected
3. Select repository: **`AbhishekItmc/dermatology-poc`**
4. Fill in:
   - **Name**: `dermatology-backend`
   - **Region**: Same as database
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Environment**: **Docker**
   - **Dockerfile Path**: `Dockerfile.render`
   - **Plan**: **Free**

5. **Add Environment Variables** (click "Advanced" or scroll down):
   ```
   POSTGRES_SERVER = <from PostgreSQL internal host>
   POSTGRES_USER = <from PostgreSQL>
   POSTGRES_PASSWORD = <from PostgreSQL>
   POSTGRES_DB = dermatology_poc
   REDIS_HOST = <from Redis internal host>
   REDIS_PORT = 6379
   ENVIRONMENT = production
   SECRET_KEY = <generate random 32 characters>
   ENCRYPTION_KEY = <generate random 32 characters>
   ```

   **How to get database values:**
   - Go to your PostgreSQL service
   - Click "Info" tab
   - Copy: Host, Username, Password

   **How to get Redis values:**
   - Go to your Redis service
   - Click "Info" tab
   - Copy: Host (internal)

6. Click **"Create Web Service"**
7. Wait 10-15 minutes for build to complete

## Step 4: Deploy Frontend

1. Click **"New"** → **"Web Service"**
2. Select repository: **`AbhishekItmc/dermatology-poc`**
3. Fill in:
   - **Name**: `dermatology-frontend`
   - **Region**: Same as backend
   - **Branch**: `main`
   - **Root Directory**: `frontend`
   - **Environment**: **Docker**
   - **Dockerfile Path**: `Dockerfile`
   - **Plan**: **Free**

4. **Add Environment Variable**:
   ```
   REACT_APP_API_URL = https://dermatology-backend.onrender.com/api/v1
   ```
   
   **Note**: Replace `dermatology-backend` with your actual backend service name if different

5. Click **"Create Web Service"**
6. Wait 10-15 minutes for build to complete

## Step 5: Access Your Application

Once all services show "Live" status:

- **Frontend**: `https://dermatology-frontend.onrender.com`
- **Backend API**: `https://dermatology-backend.onrender.com`
- **API Docs**: `https://dermatology-backend.onrender.com/api/v1/docs`

## Step 6: Test Login

Use these credentials:
- **Username**: `admin`
- **Password**: `admin123`

## Troubleshooting

### Backend won't start?
- Check logs in Render dashboard
- Verify all environment variables are set
- Make sure PostgreSQL and Redis are "Live" before backend

### Frontend can't connect to backend?
- Check the `REACT_APP_API_URL` environment variable
- Make sure it points to your backend URL
- Rebuild frontend after changing env vars

### Services spinning down?
- Free tier services sleep after 15 minutes of inactivity
- First request takes 30-60 seconds to wake up
- This is normal for free tier

## Generate Random Keys

For SECRET_KEY and ENCRYPTION_KEY, use one of these methods:

**PowerShell:**
```powershell
-join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | ForEach-Object {[char]$_})
```

**Online:**
- Go to: https://www.random.org/strings/
- Generate 32 character random string

## Important Notes

- Free tier has 750 hours/month (enough for demo)
- ML features are disabled on free tier (returns 503 error)
- To enable ML features, upgrade to paid plan ($7/month) and change Dockerfile to `Dockerfile` instead of `Dockerfile.render`
