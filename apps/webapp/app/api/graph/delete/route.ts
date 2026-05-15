import { prisma } from '@/lib/db';
import { DeleteObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { NextResponse } from 'next/server';
import { object, string, ValidationError } from 'yup';

const deleteGraphSchema = object({
  modelId: string().required(),
  slug: string().required(),
});

const s3Client = new S3Client({ region: 'us-east-1' });

/**
 * @swagger
 * /api/graph/delete:
 *   post:
 *     summary: Delete Graph
 *     description: Deletes an existing graph from Neuronpedia. You can only delete graphs you created.
 *     tags:
 *       - Attribution Graphs
 *     security:
 *       - apiKey: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - modelId
 *               - slug
 *             properties:
 *               modelId:
 *                 type: string
 *                 description: ID of the model
 *               slug:
 *                 type: string
 *                 description: Slug of the graph to delete
 *     responses:
 *       200:
 *         description: Graph deleted successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 */

export async function POST(request: Request) {
  const bodyJson = await request.json();

  try {
    const body = await deleteGraphSchema.validate(bodyJson);

    // Find the graph metadata to get the URL
    const graphMetadata = await prisma.graphMetadata.findUnique({
      where: {
        modelId_slug: {
          modelId: body.modelId,
          slug: body.slug,
        },
      },
    });

    if (!graphMetadata) {
      return NextResponse.json({ message: 'Graph not found' }, { status: 404 });
    }

    // Parse the URL to get the S3 bucket and key.
    // Supports both AWS S3 (https://bucket.s3.amazonaws.com/key) and
    // path-style endpoints like MinIO (http://host:port/bucket/key).
    const { url } = graphMetadata;
    let bucket: string | undefined;
    let s3Key = '';

    const awsMatch = url.match(/^https?:\/\/([^.]+)\.s3[^/]*\.amazonaws\.com\/(.+)$/);
    if (awsMatch) {
      [, bucket, s3Key] = awsMatch;
    } else {
      const pathStyleMatch = url.match(/^https?:\/\/[^/]+\/([^/]+)\/(.+)$/);
      if (pathStyleMatch) {
        [, bucket, s3Key] = pathStyleMatch;
      }
    }

    console.log('bucket', bucket, 's3 object to delete', s3Key);

    if (bucket && s3Key) {
      try {
        const deleteCommand = new DeleteObjectCommand({ Bucket: bucket, Key: s3Key });
        const response = await s3Client.send(deleteCommand);
        if (response.$metadata.httpStatusCode === 204) {
          console.log('S3 object deleted successfully');
        } else {
          console.warn(`S3 object deletion returned ${response.$metadata.httpStatusCode}; continuing with metadata delete`);
        }
      } catch (s3Error) {
        console.warn('S3 object deletion failed; continuing with metadata delete:', s3Error);
      }
    }

    await prisma.graphMetadata.delete({
      where: {
        modelId_slug: {
          modelId: body.modelId,
          slug: body.slug,
        },
      },
    });

    console.log('graph metadata deleted');

    return NextResponse.json({ message: 'Graph deleted successfully' });
  } catch (error) {
    if (error instanceof ValidationError) {
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    console.error('Error deleting graph:', error);
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
}
