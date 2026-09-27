"""docgen endpoints: upload a عقد, review what was extracted, render the قرار.

OCR is queued by the page-map route, not by upload: the lawyer's submitted
page map is what tells `run_ocr_job` which pages to OCR and where each value
lives. It runs as a background task with status polling rather than SSE:
unlike /chat there is nothing to stream, and the job outlives the request.
Every route resolves the session through `service.get_session`, which
filters on user_id in the query -- ownership is never checked after the fact.
"""

from __future__ import annotations

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from legal_assistant.api.deps import get_current_user_id, get_db_session
from legal_assistant.api.schemas import (
    DocgenArticleOut,
    DocgenArticlePatchRequest,
    DocgenAttendeesPatchRequest,
    DocgenFieldsPatchRequest,
    DocgenPageMapRequest,
    DocgenSessionCreateRequest,
    DocgenSessionDetailOut,
    DocgenSessionOut,
    DocgenUploadOut,
)
from legal_assistant.docgen import service
from legal_assistant.docgen.models import DocgenSession
from legal_assistant.docgen.numbering import article_name
from legal_assistant.docgen.pdf import InvalidPdfError

router = APIRouter(prefix="/docgen", tags=["docgen"])

# A scanned عقد تأسيس at moderate scanning DPI runs tens of pages; 30MB is
# comfortably above that (a few MB/page at typical scan quality) while still
# bounding how much attacker-controlled data this, the newest network-facing
# surface in the system, will buffer in memory before pdf.page_count/
# storage.write ever see it.
_MAX_UPLOAD_BYTES = 30 * 1024 * 1024


def _session_out(session: DocgenSession) -> DocgenSessionOut:
    shown_status, shown_error = service.displayed_status(session)
    return DocgenSessionOut(
        id=session.id,
        company_type=session.company_type,
        source_mode=session.source_mode,
        status=shown_status,
        error=shown_error,
        has_document=session.document_key is not None,
        created_at=session.created_at,
        updated_at=session.updated_at,
        expires_at=session.expires_at,
        page_map=session.page_map,
    )


def _article_out(a) -> DocgenArticleOut:
    return DocgenArticleOut(
        position=a.position,
        article_number=a.article_number,
        is_mukarrar=a.is_mukarrar,
        ordinal_words=a.ordinal_words,
        article_name=article_name(a.article_number, a.is_mukarrar),
        status=a.status,
        span_first=a.span_first,
        span_last=a.span_last,
        possibly_truncated=a.possibly_truncated,
        source_text=a.source_text,
        patched_text=a.patched_text,
        patch_ops=a.patch_ops,
        confidence=a.confidence,
        needs_review=a.needs_review,
        new_text=a.new_text,
    )


def _detail_out(session: DocgenSession) -> DocgenSessionDetailOut:
    return DocgenSessionDetailOut(
        **_session_out(session).model_dump(),
        uploads=[
            DocgenUploadOut(
                id=u.id,
                kind=u.kind,
                filename=u.filename,
                page_count=u.page_count,
            )
            for u in session.uploads
        ],
        articles=[_article_out(a) for a in session.articles],
        fields=(session.fields.data if session.fields else {}),
        warning=service.displayed_status(session)[1],
    )


async def _load(db: AsyncSession, session_id: int, user_id: int) -> DocgenSession:
    try:
        return await service.get_session(db, session_id, user_id)
    except service.SessionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.get("/sessions", response_model=list[DocgenSessionOut])
async def list_sessions_route(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db_session),
) -> list[DocgenSessionOut]:
    return [_session_out(s) for s in await service.list_sessions(db, user_id)]


@router.post("/sessions", response_model=DocgenSessionOut, status_code=status.HTTP_201_CREATED)
async def create_session_route(
    body: DocgenSessionCreateRequest,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db_session),
) -> DocgenSessionOut:
    try:
        session = await service.create_session(db, user_id, body.company_type, body.source_mode)
    except (KeyError, service.SessionStateError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    return _session_out(session)


@router.post(
    "/sessions/{session_id}/uploads",
    response_model=DocgenUploadOut,
    status_code=status.HTTP_201_CREATED,
)
async def upload_route(
    session_id: int,
    kind: str = Form(...),
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db_session),
) -> DocgenUploadOut:
    session = await _load(db, session_id, user_id)
    data = await file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="حجم الملف أكبر من الحد المسموح به.",
        )
    try:
        upload = await service.add_upload(db, session, kind, file.filename or "upload.pdf", data)
    except (InvalidPdfError, service.SessionStateError, service._StageError) as e:
        # `add_upload` runs `pdf.page_count` through `service._run_stage_thread`,
        # which wraps a malformed-PDF `InvalidPdfError` into `_StageError`
        # before it ever reaches this handler -- so a bad upload must be
        # caught here too, not just `InvalidPdfError` itself, or it falls
        # through to an unhandled 500. `_StageError`'s message is a static,
        # content-free Arabic string (stage name + exception TYPE only; see
        # `service._fail_stage`), so it is safe to surface verbatim.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    return DocgenUploadOut(
        id=upload.id,
        kind=upload.kind,
        filename=upload.filename,
        page_count=upload.page_count,
    )


@router.get("/sessions/{session_id}", response_model=DocgenSessionDetailOut)
async def get_session_route(
    session_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db_session),
) -> DocgenSessionDetailOut:
    return _detail_out(await _load(db, session_id, user_id))


@router.put("/sessions/{session_id}/page-map", response_model=DocgenSessionDetailOut)
async def put_page_map_route(
    session_id: int,
    body: DocgenPageMapRequest,
    background: BackgroundTasks,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db_session),
) -> DocgenSessionDetailOut:
    session = await _load(db, session_id, user_id)
    try:
        await service.submit_page_map(db, session, body.model_dump(by_alias=True))
        # Claim and commit BEFORE queuing: the job runs in its own DB session
        # and may start before `get_db_session` commits this request's. The
        # claim here also makes this response already read `ocr_running`.
        if not await service.claim_for_ocr(db, session.id):
            # Lost a race with another submission: raising rolls this
            # request back, so no page map is stored without a job for it.
            raise service.SessionStateError(service._OCR_BUSY)
        await service._safe_commit(db)
    except service.PageMapError as e:
        # Lawyer-entered page/article numbers and fixed Arabic text only.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.problems
        ) from e
    except service.SessionStateError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    background.add_task(service.run_claimed_ocr_job, session.id)
    db.expire_all()  # the claim was a Core UPDATE; reload the row
    return _detail_out(await _load(db, session_id, user_id))


@router.get("/sessions/{session_id}/uploads/{upload_id}/pages/{page}")
async def thumbnail_route(
    session_id: int,
    upload_id: int,
    page: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db_session),
) -> Response:
    session = await _load(db, session_id, user_id)
    try:
        png = await service.thumbnail(session, upload_id, page)
    except service.SessionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except service._StageError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    # The page carries national IDs: never let a browser or proxy keep it.
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.patch("/sessions/{session_id}/fields", response_model=DocgenSessionDetailOut)
async def patch_fields_route(
    session_id: int,
    body: DocgenFieldsPatchRequest,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db_session),
) -> DocgenSessionDetailOut:
    session = await _load(db, session_id, user_id)
    try:
        await service.update_fields(db, session, body.fields)
    except service.SessionStateError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    return _detail_out(await _load(db, session_id, user_id))


@router.patch(
    "/sessions/{session_id}/articles/{position}", response_model=DocgenArticleOut
)
async def patch_article_route(
    session_id: int,
    position: int,
    body: DocgenArticlePatchRequest,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db_session),
) -> DocgenArticleOut:
    session = await _load(db, session_id, user_id)
    try:
        article = await service.update_article(
            db, session, position, patched_text=body.patched_text, new_text=body.new_text
        )
    except service.SessionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except service.SessionStateError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    return _article_out(article)


@router.patch("/sessions/{session_id}/attendees", response_model=DocgenSessionDetailOut)
async def patch_attendees_route(
    session_id: int,
    body: DocgenAttendeesPatchRequest,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db_session),
) -> DocgenSessionDetailOut:
    session = await _load(db, session_id, user_id)
    try:
        await service.update_attendees(db, session, [a.model_dump() for a in body.attendees])
    except service.SessionStateError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    return _detail_out(await _load(db, session_id, user_id))


@router.post("/sessions/{session_id}/render", response_model=DocgenSessionOut)
async def render_route(
    session_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db_session),
) -> DocgenSessionOut:
    session = await _load(db, session_id, user_id)
    try:
        await service.render_session(db, session)
    except service.SessionStateError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    return _session_out(session)


@router.get("/sessions/{session_id}/document")
async def download_route(
    session_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db_session),
) -> Response:
    session = await _load(db, session_id, user_id)
    if not session.document_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="لم يتم إنشاء المستند بعد، أو تم تعديل البيانات بعد إنشائه.",
        )
    try:
        content = await service.read_document(session)
    except service._StageError as e:
        # Content-free by construction (stage name + exception type).
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    return Response(
        content=content,
        media_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        headers={
            "Content-Disposition": f'attachment; filename="docgen-{session.id}.docx"'
        },
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session_route(
    session_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db_session),
) -> Response:
    session = await _load(db, session_id, user_id)
    await service.delete_session(db, session)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
