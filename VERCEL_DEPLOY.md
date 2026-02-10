# Deploy to Vercel (Easiest Way!)

## Quick Deploy - Frontend Only (3 minutes)

This will deploy just the frontend UI so you can show the interface to your client.

### Step 1: Sign Up for Vercel (1 minute)

1. Go to: **https://vercel.com**
2. Click **"Sign Up"**
3. Choose **"Continue with GitHub"**
4. Authorize Vercel to access your GitHub

### Step 2: Deploy (2 minutes)

1. Click **"Add New..."** → **"Project"**
2. Find and select: **`AbhishekItmc/dermatology-poc`**
3. Click **"Import"**
4. Configure:
   - **Framework Preset**: Create React App
   - **Root Directory**: `frontend` ← Click "Edit" and type this
   - **Build Command**: `npm run build` (should auto-fill)
   - **Output Directory**: `build` (should auto-fill)
5. Click **"Deploy"**
6. Wait 2-3 minutes

### Step 3: Access Your App!

Once deployment completes:
- You'll get a URL like: `https://dermatology-poc-xyz.vercel.app`
- Click it to see your frontend!

### What Works:
- ✅ Full UI/UX visible
- ✅ All components render
- ✅ 3D viewer displays
- ✅ Perfect for showing interface to client

### What Doesn't Work (Frontend Only):
- ❌ Login (no backend)
- ❌ Data saving (no database)
- ❌ API calls (no backend)

This is perfect for demonstrating the UI/interface!

---

## Full Stack Deploy (Frontend + Backend)

If you need the backend too, Vercel can't run Python easily. Instead:

### Option A: Frontend on Vercel + Backend on Render

1. Deploy frontend to Vercel (steps above)
2. Deploy backend to Render (use existing database)
3. Update Vercel environment variable:
   - Go to Vercel project settings
   - Add: `REACT_APP_API_URL` = `https://your-backend.onrender.com/api/v1`
   - Redeploy

### Option B: Use Railway for Everything

Railway handles both frontend and backend automatically:
1. Go to: https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub"
4. Select your repo
5. Railway auto-detects and deploys everything
6. Done!

---

## Recommended Approach

**For Client Demo:**
- Use Vercel (frontend only) - Shows the beautiful UI instantly

**For Full Functionality:**
- Use Railway - Handles everything automatically

---

## Troubleshooting

### Vercel build fails?
- Make sure Root Directory is set to `frontend`
- Check that `package.json` exists in frontend folder

### Want to add backend later?
- Deploy backend to Render
- Add `REACT_APP_API_URL` environment variable in Vercel
- Redeploy frontend

---

## Next Steps

After deploying to Vercel:
1. Share the URL with your client
2. They can see the full interface
3. If they want functionality, deploy backend to Render or Railway
