const CANONICAL_HOST = "musclememory.space";
const DAYTONA_ORIGIN =
  "https://8000-af37e210-f22a-4e83-a7b3-b3d97eed10ee.proxy.daytona.work";

export default {
  async fetch(request) {
    const publicUrl = new URL(request.url);
    if (publicUrl.hostname === `www.${CANONICAL_HOST}`) {
      publicUrl.hostname = CANONICAL_HOST;
      return Response.redirect(publicUrl.toString(), 308);
    }
    if (publicUrl.hostname !== CANONICAL_HOST) {
      return new Response("Not found", { status: 404 });
    }

    const upstreamUrl = new URL(DAYTONA_ORIGIN);
    upstreamUrl.pathname = publicUrl.pathname;
    upstreamUrl.search = publicUrl.search;

    const upstreamRequest = new Request(upstreamUrl.toString(), request);
    const headers = new Headers(upstreamRequest.headers);
    headers.set("X-Daytona-Skip-Preview-Warning", "true");
    headers.set("X-Daytona-Disable-CORS", "true");
    headers.set("X-Forwarded-Host", publicUrl.host);
    headers.set("X-Forwarded-Proto", "https");

    const response = await fetch(new Request(upstreamRequest, { headers }));
    if (response.status === 101) {
      return response;
    }

    const responseHeaders = new Headers(response.headers);
    const location = responseHeaders.get("Location");
    if (location) {
      const redirectUrl = new URL(location, upstreamUrl);
      if (redirectUrl.origin === new URL(DAYTONA_ORIGIN).origin) {
        redirectUrl.protocol = "https:";
        redirectUrl.host = CANONICAL_HOST;
        responseHeaders.set("Location", redirectUrl.toString());
      }
    }
    if (publicUrl.pathname.startsWith("/api/")) {
      responseHeaders.set("Cache-Control", "no-store");
    }

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: responseHeaders,
    });
  },
};
