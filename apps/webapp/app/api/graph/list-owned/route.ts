import { prisma } from '@/lib/db';
import { NextResponse } from 'next/server';

/**
 * @swagger
 * /api/graph/list:
 *   get:
 *     summary: List User's Graphs
 *     description: Retrieves a list of all graph metadata for the authenticated user
 *     tags:
 *       - Attribution Graphs
 *     security:
 *       - apiKey: []
 *     responses:
 *       200:
 *         description: Successfully retrieved graph list
 */

export async function POST() {
  try {
    const graphMetadatas = await prisma.graphMetadata.findMany({
      orderBy: {
        updatedAt: 'desc',
      },
    });

    return NextResponse.json(graphMetadatas);
  } catch (error) {
    console.error('Error fetching graph list:', error);
    return NextResponse.json({ error: 'Failed to fetch graph list' }, { status: 500 });
  }
}
