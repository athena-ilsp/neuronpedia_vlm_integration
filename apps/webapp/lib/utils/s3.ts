import { AWS_ACCESS_KEY_ID, AWS_ENDPOINT_URL, AWS_REGION, AWS_SECRET_ACCESS_KEY, NEXT_PUBLIC_URL } from '@/lib/env';
import { GetObjectCommand, ListObjectsV2Command, S3, S3Client } from '@aws-sdk/client-s3';
import { ungzip } from 'node-gzip';
import { getLayerNumFromSource, getSourceSetNameFromSource } from './source';

export const DATASET_BUCKET = 'neuronpedia-datasets';
export const DATASET_REGION = 'us-east-1';
export const DATASET_VERSION = 'v1';
export const DATASET_BASE_PATH = `${DATASET_VERSION}/`;
export const CONFIG_DIR_NAME = 'config';
export const CONFIG_BASE_PATH = `${DATASET_BASE_PATH}${CONFIG_DIR_NAME}/`;
export const anonymousS3Client = new S3({
  region: DATASET_REGION,
  credentials: { accessKeyId: '', secretAccessKey: '' },
  signer: { sign: async (req) => req },
});

/**
 * The credentialed client user uploads (graphs) are written with. With `AWS_ENDPOINT_URL` set it
 * talks to that S3-compatible store instead of AWS, with path-style URLs as MinIO needs, so a
 * presigned URL and the object's public URL are both `<endpoint>/<bucket>/<key>`.
 */
export function makeUploadS3Client() {
  return new S3Client({
    region: AWS_REGION,
    credentials: { accessKeyId: AWS_ACCESS_KEY_ID, secretAccessKey: AWS_SECRET_ACCESS_KEY },
    ...(AWS_ENDPOINT_URL ? { endpoint: AWS_ENDPOINT_URL, forcePathStyle: true } : {}),
  });
}

// next.config.js proxies this path to AWS_ENDPOINT_URL; keep the two in sync.
export const GRAPH_STORAGE_PROXY_PATH = '/graph-storage';

/**
 * The URL to store for an uploaded object, which browsers will fetch. An object on an
 * S3-compatible store (`AWS_ENDPOINT_URL`) is served through the webapp's own origin, so a
 * browser on another machine needs no route to the store; an AWS URL is returned as is.
 */
export function browserUrlForUpload(objectUrl: string): string {
  if (!AWS_ENDPOINT_URL || !objectUrl.startsWith(`${AWS_ENDPOINT_URL}/`)) {
    return objectUrl;
  }
  return `${NEXT_PUBLIC_URL}${GRAPH_STORAGE_PROXY_PATH}/${objectUrl.slice(AWS_ENDPOINT_URL.length + 1)}`;
}

async function downloadFile(path: string) {
  const response = await anonymousS3Client.send(
    new GetObjectCommand({
      Bucket: DATASET_BUCKET,
      Key: path,
    }),
  );
  return response.Body;
}

export async function listBucketAtPath(path: string, delimiter?: string) {
  const response = await anonymousS3Client.send(
    new ListObjectsV2Command({
      Bucket: DATASET_BUCKET,
      Prefix: path,
      Delimiter: delimiter,
    }),
  );
  return response;
}

export function parseLine(line: string) {
  const parsedData = JSON.parse(line);
  if (parsedData.createdAt) {
    parsedData.createdAt = new Date(parsedData.createdAt);
  }
  if (parsedData.updatedAt) {
    parsedData.updatedAt = new Date(parsedData.updatedAt);
  }
  return parsedData;
}

export async function getS3ModelsToSources() {
  const s3ModelsResponse = await listBucketAtPath(DATASET_BASE_PATH, '/');
  const s3models = s3ModelsResponse.CommonPrefixes?.map((content) => content.Prefix || '') || [];

  const s3modelToSources: Record<string, string[]> = Object.fromEntries(
    await Promise.all(
      s3models.map(async (model) => {
        const response = await listBucketAtPath(model, '/');
        return [
          model.replace(DATASET_BASE_PATH, '').replace('/', ''),
          response.CommonPrefixes?.map((content) => content.Prefix?.split('/')[2]) || [],
        ];
      }),
    ),
  );
  Object.keys(s3modelToSources).forEach((model) => {
    if (model === CONFIG_DIR_NAME) {
      delete s3modelToSources[model];
    }
  });
  // Sort sources for each model
  Object.keys(s3modelToSources).forEach((model) => {
    s3modelToSources[model].sort((a, b) => {
      const sourceSetA = getSourceSetNameFromSource(a || '');
      const sourceSetB = getSourceSetNameFromSource(b || '');
      if (sourceSetA !== sourceSetB) {
        return sourceSetA.localeCompare(sourceSetB);
      }
      return getLayerNumFromSource(a || '') - getLayerNumFromSource(b || '');
    });
  });
  return s3modelToSources;
}

export async function getFilesInPath(
  path: string,
  filterByFileSuffix?: string,
  includeSubdirectories: boolean = true,
): Promise<string[]> {
  const response = await anonymousS3Client.send(
    new ListObjectsV2Command({
      Bucket: DATASET_BUCKET,
      Prefix: path,
    }),
  );
  const files =
    response.Contents?.map((content) => content.Key || '')
      .filter(Boolean)
      .filter((file) => {
        if (includeSubdirectories) return true;
        // Extract the part of the path after the prefix, truncating the first slash. eg "v1/activations/batch-1.json.gz" -> "/batch-1.json.gz" -> "batch-1.json.jz"
        const relativePath = file.substring(path.length).substring(1);
        // if it still has a slash, it's in a subdirectory
        return !relativePath.includes('/');
      })
      .filter((file) => (filterByFileSuffix ? file.endsWith(filterByFileSuffix) : true)) || [];
  console.log('Files in path', files);
  return files;
}

export async function downloadAndDecompressFile(path: string): Promise<string> {
  console.log('Downloading and decompressing file', path);
  const file = await downloadFile(path);
  const arrayBuffer = await file?.transformToByteArray();
  return arrayBuffer ? (await ungzip(Buffer.from(arrayBuffer))).toString() : '';
}

async function downloadFileAsString(path: string): Promise<string> {
  const file = await downloadFile(path);
  if (!file) {
    throw new Error('File not found');
  }
  return file?.transformToString() || '';
}

export async function downloadFileJsonlParsedLines(path: string): Promise<any[]> {
  const file = await downloadFileAsString(path);
  return file.split('\n').filter(Boolean).map(parseLine);
}
