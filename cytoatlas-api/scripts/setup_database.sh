#!/bin/bash
# Set up PostgreSQL database for CytoAtlas API

set -e

# Configuration
DB_NAME="${DB_NAME:-cytoatlas}"
DB_USER="${DB_USER:-cytoatlas}"
DB_PASSWORD="${DB_PASSWORD:-cytoatlas}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=========================================="
echo "  CytoAtlas Database Setup"
echo "=========================================="

# Check if psql is available
if ! command -v psql &> /dev/null; then
    echo "Error: psql command not found"
    echo "Please ensure PostgreSQL client is installed"
    exit 1
fi

# Check connection
echo "Checking PostgreSQL connection..."
if ! psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -c "SELECT 1" &> /dev/null; then
    echo "Warning: Cannot connect to PostgreSQL as postgres user"
    echo "You may need to create the database manually"
fi

# Create database and user (if admin access available)
echo ""
echo "Creating database and user..."
echo "(You may be prompted for postgres password)"

psql -h "$DB_HOST" -p "$DB_PORT" -U postgres << EOF || true
-- Create user if not exists
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$DB_USER') THEN
        CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';
    END IF;
END
\$\$;

-- Create database if not exists
SELECT 'CREATE DATABASE $DB_NAME OWNER $DB_USER'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$DB_NAME')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
EOF

echo ""
echo "Database setup complete!"
echo ""
echo "Connection string:"
echo "  postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
echo ""

# Activate conda and run migrations
echo "Running database migrations..."
cd "$PROJECT_DIR"

source ~/bin/myconda 2>/dev/null || true
conda activate secactpy 2>/dev/null || true

# Export DATABASE_URL for alembic
export DATABASE_URL="postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

# Run alembic migrations
if command -v alembic &> /dev/null; then
    alembic upgrade head
    echo "Migrations complete!"
else
    echo "Alembic not found. Install with: pip install alembic"
    echo "Then run: alembic upgrade head"
fi
