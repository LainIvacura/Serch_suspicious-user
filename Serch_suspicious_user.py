from fastapi import FastAPI, HTTPException, Depends, Query
from suppadel import ConnectionFactory
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

from config import (
    TARGET_DB_NAME, USER_DB_NAME, CLASSIFIER_DB_NAME, LOG_DB_NAME, ORG_DB_NAME,
    INTERNAL_IP_PREFIX,
    GOODS_URI_PATTERN, KTRU_URI_PATTERN, PROPOSAL_URI_PATTERN,
    HOST, PORT,
)

app = FastAPI(title="Customer Management API")


# =====================================================================
# Pydantic-модели ответов
# =====================================================================

class UserInfo(BaseModel):
    """Информация об одном пользователе-заказчике (ответ /search-by-inn)."""
    userId: int
    username: str
    lastName: str
    firstName: str
    middleName: str
    organizationId: int
    type: str
    inn: str
    shortName: str
    kpp: str
    ips: List[str]
    is_blocked: Optional[bool] = None
    block_status: Optional[str] = None


class SearchByINNResponse(BaseModel):
    """Ответ эндпоинта /search-by-inn."""
    success: bool
    message: str
    customer_info: Optional[Dict[str, Any]] = None
    users_with_ips: List[UserInfo] = []


class KTRUUserInfo(BaseModel):
    """Информация об одном пользователе (ответ /search-by-ktru)."""
    userId: int
    username: str
    lastName: str
    firstName: str
    middleName: str
    unique_remote_addrs: List[str]
    view_times: List[str] = []
    organization_info: Optional[Dict[str, Any]] = None
    is_blocked: Optional[bool] = None
    block_status: Optional[str] = None


class SearchByKTRUResponse(BaseModel):
    """Ответ эндпоинта /search-by-ktru."""
    success: bool
    message: str
    ktru_info: Optional[Dict[str, Any]] = None
    users: List[KTRUUserInfo] = []


class SearchProposalsResponse(BaseModel):
    """Ответ эндпоинта /search-proposals."""
    success: bool
    message: str
    view_proposal: List[int] = []
    users: List[Dict[str, Any]] = []
    search_criteria: Optional[Dict[str, Any]] = None


# =====================================================================
# Подключения к источникам данных
# =====================================================================

def get_target_db_connection():
    """
    Соединение с целевой БД сервиса .
    Используется для чтения данных о предложениях и статусах блокировки.
    """
    cf = ConnectionFactory()
    return cf.get_connection(TARGET_DB_NAME, prod=True)


def get_user_db_connection():
    """
    Соединение с БД пользователей и организаций . Нужна для получения ФИО, ИНН, типа организации
    и поиска пользователей по ИНН.
    """
    cf = ConnectionFactory()
    return cf.get_connection(USER_DB_NAME, prod=True)


def get_log_db_connection():
    """
    Соединение с ClickHouse, где лежат http-логи.
    Используется для поиска IP-адресов и времени просмотров по URI.
    """
    cf = ConnectionFactory()
    return cf.get_clickhouse_connection(LOG_DB_NAME)


def get_classifier_db_connection():
    """
    Соединение с БД классификатора.
    По коду КТРУ возвращает его id и наименование.
    """
    cf = ConnectionFactory()
    return cf.get_connection(CLASSIFIER_DB_NAME)


def get_org_db_connection():
    """
    Соединение с БД организаций (таблица organization).
    Используется в /search-proposals для поиска организации по ИНН.
    """
    cf = ConnectionFactory()
    return cf.get_connection(ORG_DB_NAME, prod=True)


# =====================================================================
# Вспомогательные функции
# =====================================================================

def check_block_status_batch(user_connection, target_connection, users_data: List[Dict[str, Any]]) -> Dict[
    str, Dict[str, Any]]:
    """
    Пакетная проверка статуса блокировки пользователей.

    Принимает список словарей вида [{"username": ..., "inn": ...}, ...].
    Одним запросом получает GUID'ы пользователей из user-БД, затем одним
    запросом тянет записи из blockCustomers целевой БД.

    Возвращает словарь {username: {"is_blocked": bool|None, "block_status": str}},
    где block_status может быть:
      - "никогда не блокировался"
      - "заблокирован"
      - "разблокирован (был заблокирован ранее)"
      - "пользователь не найден в системе"
      - "ошибка проверки: ..."
    """
    result = {}
    if not users_data:
        return result

    try:
        usernames = [user["username"] for user in users_data]

        with user_connection.get_session() as user_sess:
            user_table = user_connection.get_table('user')
            user_guid_results = user_sess.query(
                user_table.username,
                user_table.guid
            ).filter(
                user_table.username.in_(usernames),
                user_table.active == 1
            ).all()
            username_to_guid = {row[0]: row[1] for row in user_guid_results}

        with target_connection.get_session() as target_sess:
            block_table = target_connection.get_table('blockCustomers')
            for user_data in users_data:
                username = user_data["username"]
                inn = user_data["inn"]
                user_guid = username_to_guid.get(username)

                if not user_guid:
                    result[username] = {"is_blocked": None, "block_status": "пользователь не найден в системе"}
                    continue

                records = target_sess.query(block_table.active).filter(
                    block_table.customerInn == inn,
                    block_table.userGuid == user_guid
                ).all()

                if not records:
                    result[username] = {"is_blocked": False, "block_status": "никогда не блокировался"}
                elif any(record[0] == 1 for record in records):
                    result[username] = {"is_blocked": True, "block_status": "заблокирован"}
                else:
                    result[username] = {"is_blocked": False, "block_status": "разблокирован (был заблокирован ранее)"}

    except Exception as e:
        for user_data in users_data:
            result[user_data["username"]] = {"is_blocked": None, "block_status": f"ошибка проверки: {str(e)}"}

    return result


def get_user_organization_info(user_connection, user_id: int) -> Optional[Dict[str, Any]]:
    """
    Полная информация о пользователе и его организации по user_id.

    Делает JOIN user → organizationMember → organization и возвращает
    ФИО, username, ИНН, тип организации, shortName, kpp.
    Возвращает None, если пользователь не найден или неактивен.
    """
    try:
        with user_connection.get_session() as sess:
            org_table = user_connection.get_table('organization')
            org_member_table = user_connection.get_table('organizationMember')
            user_table = user_connection.get_table('user')

            result = sess.query(
                user_table.username,
                user_table.lastName,
                user_table.firstName,
                user_table.middleName,
                org_table.id.label('organization_id'),
                org_table.type,
                org_table.inn,
                org_table.shortName,
                org_table.kpp
            ).join(
                org_member_table, org_table.id == org_member_table.organizationId
            ).join(
                user_table, org_member_table.userId == user_table.id
            ).filter(
                user_table.id == user_id,
                org_table.active == 1,
                user_table.active == 1
            ).first()

            if result:
                middle_name = result.middleName
                if middle_name is None:
                    middle_name = ""

                return {
                    "username": result.username,
                    "lastName": result.lastName,
                    "firstName": result.firstName,
                    "middleName": middle_name,
                    "organizationId": result.organization_id,
                    "organizationType": result.type,
                    "inn": result.inn,
                    "shortName": result.shortName,
                    "kpp": result.kpp
                }
    except Exception as e:
        print(f"Ошибка получения информации о пользователе {user_id}: {e}")

    return None


# =====================================================================
# ЭНДПОИНТ /search-by-inn
# =====================================================================

@app.get("/search-by-inn", response_model=SearchByINNResponse)
async def search_by_inn(
        inn: str = Query(...),
        months_back: int = Query(...),
        user_connection=Depends(get_user_db_connection),
        target_connection=Depends(get_target_db_connection),
        log_connection=Depends(get_log_db_connection)
):
    """
    Поиск пользователей-заказчиков по ИНН с проверкой блокировок.

    Алгоритм:
      1. По ИНН находит всех активных пользователей организации.
      2. Через ClickHouse получает список уникальных IP, с которых они
         работали за последние N месяцев (внутренняя подсеть отбрасывается).
      3. Ищет других пользователей, которые заходили с этих же IP
         по URI-паттерну GOODS_URI_PATTERN.
      4. Оставляет только тех, кто относится к организациям типа "customer".
      5. Пакетно проверяет статус блокировки каждого.

    Параметры:
      - inn: ИНН организации (только цифры)
      - months_back: глубина поиска в месяцах

    Возвращает:
      - customer_info — данные организации
      - users_with_ips — список найденных заказчиков с IP и статусом блокировки
    """

    if not inn or not inn.strip():
        raise HTTPException(status_code=400, detail="Не указан ИНН")

    inn = inn.strip()

    if not inn.isdigit():
        raise HTTPException(status_code=400, detail="ИНН должен содержать только цифры")

    try:
        # ШАГ 1: ищем всех активных пользователей организации по ИНН
        with user_connection.get_session() as sess:
            org_table = user_connection.get_table('organization')
            org_member_table = user_connection.get_table('organizationMember')
            user_table = user_connection.get_table('user')

            results = sess.query(
                user_table.id,
                user_table.username,
                user_table.lastName,
                user_table.firstName,
                user_table.middleName,
                user_table.source,
                org_table.id.label('organization_id'),
                org_table.type,
                org_table.inn,
                org_table.shortName,
                org_table.kpp
            ).join(
                org_member_table, org_table.id == org_member_table.organizationId
            ).join(
                user_table, org_member_table.userId == user_table.id
            ).filter(
                org_table.inn == inn,
                org_table.active == 1,
                user_table.active == 1
            ).all()

            if not results:
                return SearchByINNResponse(
                    success=False,
                    message=f"Не найдено активных пользователей для организации с ИНН '{inn}'."
                )

            customer_info = {
                "inn": results[0].inn,
                "shortName": results[0].shortName,
                "kpp": results[0].kpp,
                "type": results[0].type
            }
            user_ids = [row.id for row in results]

        # ШАГ 2: собираем уникальные IP пользователей за N месяцев (ClickHouse)
        user_ids_str = ', '.join(str(uid) for uid in user_ids)
        ip_query = f"""
            SELECT groupArray(DISTINCT remoteAddr) AS unique_ips
            FROM httpActions
            WHERE userId IN ({user_ids_str})
              AND dateTime >= subtractMonths(NOW(), {months_back})
              AND NOT startsWith(remoteAddr, '{INTERNAL_IP_PREFIX}')
        """
        ip_result = log_connection.connect.query(ip_query)

        if not ip_result.result_rows or not ip_result.first_row[0]:
            return SearchByINNResponse(
                success=False,
                message=f"За последние {months_back} месяц(ев) не найдено IP адресов.",
                customer_info=customer_info
            )

        ip_list = ip_result.first_row[0]

        if isinstance(ip_list, str):
            ip_list = [ip.strip("'") for ip in ip_list.split(", ")]

        if not ip_list:
            return SearchByINNResponse(
                success=False,
                message="Найдены IP адреса, но все они из внутренней сети и были отфильтрованы.",
                customer_info=customer_info
            )

        # ШАГ 3: ищем всех, кто ходил с этих IP по целевому URI
        ip_list_str = ', '.join(f"'{ip}'" for ip in ip_list)
        users_by_ip_query = f"""
            SELECT DISTINCT userId, groupArray(DISTINCT remoteAddr) AS unique_remote_addrs
            FROM httpActions 
            WHERE dateTime >= subtractMonths(now(), {months_back})
                AND uri LIKE '{GOODS_URI_PATTERN}'
                AND remoteAddr IN ({ip_list_str})
                AND userId != 1
            GROUP BY userId
            ORDER BY userId
        """
        users_by_ip_result = log_connection.connect.query(users_by_ip_query)

        if not users_by_ip_result.result_rows:
            return SearchByINNResponse(
                success=False,
                message="Не найдено пользователей, которые заходили с найденных IP адресов.",
                customer_info=customer_info
            )

        found_users = []
        for row in users_by_ip_result.result_rows:
            ips = row[1]
            if not isinstance(ips, list):
                ips = [ips]
            found_users.append({"userId": row[0], "ips": ips})

        found_user_ids = [user["userId"] for user in found_users]

        if found_user_ids:
            # ШАГ 4: обогащаем данными об организациях и оставляем только customer
            with user_connection.get_session() as final_sess:
                org_table = user_connection.get_table('organization')
                org_member_table = user_connection.get_table('organizationMember')
                user_table = user_connection.get_table('user')

                final_results = final_sess.query(
                    user_table.id,
                    user_table.username,
                    user_table.lastName,
                    user_table.firstName,
                    user_table.middleName,
                    org_table.id.label('organization_id'),
                    org_table.type,
                    org_table.inn,
                    org_table.shortName,
                    org_table.kpp
                ).join(
                    org_member_table, org_table.id == org_member_table.organizationId
                ).join(
                    user_table, org_member_table.userId == user_table.id
                ).filter(
                    user_table.id.in_(found_user_ids),
                    org_table.active == 1,
                    user_table.active == 1,
                    org_table.type == 'customer'
                ).all()

                if not final_results:
                    return SearchByINNResponse(
                        success=False,
                        message=f"Найдено {len(found_user_ids)} пользователей, но ни один не является заказчиком.",
                        customer_info=customer_info
                    )

                # ШАГ 5: пакетная проверка блокировок
                users_for_block_check = []
                for user in final_results:
                    users_for_block_check.append({"username": user.username, "inn": user.inn})

                block_statuses = check_block_status_batch(user_connection, target_connection, users_for_block_check)
                result_users = []

                for user in final_results:
                    user_ips = []
                    for u in found_users:
                        if u["userId"] == user.id:
                            user_ips = u["ips"]
                            break

                    block_info = block_statuses.get(user.username,
                                                    {"is_blocked": None, "block_status": "статус не определен"})

                    middle_name = user.middleName
                    if not hasattr(user, 'middleName') or middle_name is None:
                        middle_name = ""

                    ips_list = user_ips
                    if not isinstance(user_ips, list):
                        ips_list = [user_ips]

                    result_users.append(UserInfo(
                        userId=user.id,
                        username=user.username,
                        lastName=user.lastName,
                        firstName=user.firstName,
                        middleName=middle_name,
                        organizationId=user.organization_id,
                        type=user.type,
                        inn=user.inn,
                        shortName=user.shortName,
                        kpp=user.kpp,
                        ips=ips_list,
                        is_blocked=block_info.get("is_blocked"),
                        block_status=block_info.get("block_status")
                    ))

                return SearchByINNResponse(
                    success=True,
                    message=f"Найдено {len(result_users)} пользователей-заказчиков",
                    customer_info=customer_info,
                    users_with_ips=result_users
                )
        else:
            return SearchByINNResponse(
                success=False,
                message="Не найдено идентификаторов пользователей.",
                customer_info=customer_info
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


# =====================================================================
# ЭНДПОИНТ /search-by-ktru
# =====================================================================

@app.get("/search-by-ktru", response_model=SearchByKTRUResponse)
async def search_by_ktru(
        ktru_code: str = Query(...),
        date_from: str = Query(...),
        date_to: str = Query(...),
        user_connection=Depends(get_user_db_connection),
        classifier_connection=Depends(get_classifier_db_connection),
        target_connection=Depends(get_target_db_connection),
        log_connection=Depends(get_log_db_connection)
):
    """
    Поиск пользователей, просматривавших конкретный КТРУ за период.

    Алгоритм:
      1. По коду КТРУ находит его id в классификаторе.
      2. Через ClickHouse ищет http-запросы к каталогу по этому КТРУ
         (code = 200, userId != 1).
      3. Для каждого пользователя собирает уникальные IP и время просмотров.
      4. Обогащает данными об организации и статусом блокировки.

    Параметры:
      - ktru_code: код КТРУ
      - date_from / date_to: диапазон в формате YYYY-MM-DD HH:MM:SS

    Возвращает:
      - ktru_info — данные о КТРУ
      - users — список пользователей с IP, временем просмотров, организацией и блокировкой
    """

    if not ktru_code or not ktru_code.strip():
        raise HTTPException(status_code=400, detail="Не указан код КТРУ")

    ktru_code = ktru_code.strip()

    try:
        start_date = datetime.strptime(date_from, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(date_to, "%Y-%m-%d %H:%M:%S")
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="Дата начала не может быть позже даты окончания")
    except ValueError:
        raise HTTPException(status_code=400, detail="Неверный формат даты. Используйте: YYYY-MM-DD HH:MM:SS")

    try:
        # ШАГ 1: находим КТРУ в классификаторе
        with classifier_connection.get_session() as classifier_sess:
            ktru_table = classifier_connection.get_table('ktru')
            ktru_info = classifier_sess.query(ktru_table).filter(
                ktru_table.code == ktru_code,
                ktru_table.actual == 1
            ).first()

            if not ktru_info:
                return SearchByKTRUResponse(
                    success=False,
                    message=f"КТРУ с кодом '{ktru_code}' не найден или неактуален."
                )

            ktru_data = {
                "id": ktru_info.id,
                "code": ktru_info.code,
                "name": getattr(ktru_info, 'name', None),
                "actual": ktru_info.actual
            }

        # ШАГ 2: ищем http-запросы по этому КТРУ (ClickHouse)
        ktru_id = ktru_info.id
        search_uri = KTRU_URI_PATTERN.format(ktru_id=ktru_id)

        query = f"""
            SELECT 
                ha.userId,
                groupArrayDistinct(ha.remoteAddr) AS unique_remote_addrs,
                groupArray(toString(ha.dateTime)) AS view_times
            FROM 
                httpActions ha 
            WHERE 
                dateTime BETWEEN '{date_from}' AND '{date_to}'
                AND ha.uri LIKE '{search_uri}'
                AND ha.code = 200
                AND ha.userId != 1
            GROUP BY 
                ha.userId
            ORDER BY 
                ha.userId
        """
        log_result = log_connection.connect.query(query)

        if not log_result.result_rows:
            return SearchByKTRUResponse(
                success=False,
                message=f"Не найдено пользователей, просматривавших КТРУ '{ktru_code}' за указанный период.",
                ktru_info=ktru_data
            )

        # ШАГ 3: обогащаем каждого пользователя данными об организации
        users_data = []
        users_for_block_check = []

        for row in log_result.result_rows:
            user_id = row[0]
            unique_remote_addrs = row[1]
            view_times = row[2]

            if not isinstance(unique_remote_addrs, list):
                if unique_remote_addrs:
                    unique_remote_addrs = [unique_remote_addrs]
                else:
                    unique_remote_addrs = []

            if not isinstance(view_times, list):
                if view_times:
                    view_times = [view_times]
                else:
                    view_times = []

            # нормализуем время из ISO-формата в "YYYY-MM-DD HH:MM:SS"
            formatted_view_times = []
            for vt in view_times:
                try:
                    if isinstance(vt, str) and 'T' in vt:
                        vt_clean = vt.replace('Z', '').split('+')[0]
                        dt = datetime.fromisoformat(vt_clean)
                        formatted_view_times.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
                    else:
                        formatted_view_times.append(str(vt))
                except Exception:
                    formatted_view_times.append(str(vt))

            if formatted_view_times:
                final_view_times = formatted_view_times
            else:
                final_view_times = view_times

            org_info = get_user_organization_info(user_connection, user_id)

            if org_info:
                user_info = {
                    "user_id": user_id,
                    "username": org_info["username"],
                    "lastName": org_info["lastName"],
                    "firstName": org_info["firstName"],
                    "middleName": org_info["middleName"],
                    "inn": org_info["inn"],
                    "unique_remote_addrs": unique_remote_addrs,
                    "view_times": final_view_times,
                    "organization_info": {
                        "organizationId": org_info["organizationId"],
                        "organizationType": org_info["organizationType"],
                        "inn": org_info["inn"],
                        "shortName": org_info["shortName"],
                        "kpp": org_info["kpp"]
                    }
                }
                users_data.append(user_info)
                users_for_block_check.append({"username": org_info["username"], "inn": org_info["inn"]})
            else:
                # пользователь есть в логах, но не найден в БД — отдаём как есть
                user_info = {
                    "user_id": user_id,
                    "username": f"user_{user_id}",
                    "lastName": "",
                    "firstName": "",
                    "middleName": "",
                    "inn": None,
                    "unique_remote_addrs": unique_remote_addrs,
                    "view_times": final_view_times,
                    "organization_info": None
                }
                users_data.append(user_info)

        # ШАГ 4: пакетная проверка блокировок
        block_statuses = check_block_status_batch(user_connection, target_connection, users_for_block_check)
        users_result = []

        for user_data in users_data:
            block_info = block_statuses.get(user_data["username"])
            if block_info is None:
                block_info = {"is_blocked": None, "block_status": "статус не определен"}

            users_result.append(KTRUUserInfo(
                userId=user_data["user_id"],
                username=user_data["username"],
                lastName=user_data["lastName"],
                firstName=user_data["firstName"],
                middleName=user_data["middleName"],
                unique_remote_addrs=user_data["unique_remote_addrs"],
                view_times=user_data.get("view_times", []),
                organization_info=user_data["organization_info"],
                is_blocked=block_info.get("is_blocked"),
                block_status=block_info.get("block_status")
            ))

        return SearchByKTRUResponse(
            success=True,
            message=f"Найдено {len(users_result)} пользователей, просматривавших КТРУ '{ktru_code}'",
            ktru_info=ktru_data,
            users=users_result
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


# =====================================================================
# ЭНДПОИНТ /search-proposals
# =====================================================================

@app.get("/search-proposals", response_model=SearchProposalsResponse)
async def search_proposals(
        inn: str = Query(...),
        ktru_code: str = Query(...),
        date_from: str = Query(...),
        date_to: str = Query(...),
        user_connection=Depends(get_user_db_connection),
        target_connection=Depends(get_target_db_connection),
        classifier_connection=Depends(get_classifier_db_connection),
        log_connection=Depends(get_log_db_connection)
):
    """
    Поиск пользователей, просматривавших предварительные предложения (ПП)
    конкретной организации по конкретному КТРУ за период.

    Алгоритм:
      1. По ИНН находит организацию.
      2. По коду КТРУ находит его id в классификаторе.
      3. В целевой БД находит все proposal_information этой организации,
         затем все proposal по этому КТРУ и с updated_at <= date_to.
      4. По каждому proposal через ClickHouse ищет просмотры.
      5. Агрегирует в разрезе пользователя: какие ПП смотрел, с каких IP, когда.
      6. Обогащает данными об организации и статусом блокировки.

    Параметры:
      - inn: ИНН организации (поставщика)
      - ktru_code: код КТРУ
      - date_from / date_to: диапазон в формате YYYY-MM-DD HH:MM:SS

    Возвращает:
      - view_proposal — список ID всех найденных ПП
      - users — список пользователей с перечнем просмотренных ПП, IP и временем
      - search_criteria — эхо входных параметров
    """

    if not inn or not inn.strip():
        raise HTTPException(status_code=400, detail="Не указан ИНН")

    if not ktru_code or not ktru_code.strip():
        raise HTTPException(status_code=400, detail="Не указан код КТРУ")

    if not date_from or not date_to:
        raise HTTPException(status_code=400, detail="Не указаны даты")

    inn = inn.strip()
    ktru_code = ktru_code.strip()

    try:
        start_date = datetime.strptime(date_from, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(date_to, "%Y-%m-%d %H:%M:%S")
        if start_date > end_date:
            raise HTTPException(400, "Дата начала не может быть позже даты окончания")
    except ValueError:
        raise HTTPException(400, "Неверный формат даты. Используйте: YYYY-MM-DD HH:MM:SS")

    if not inn.isdigit():
        raise HTTPException(400, "ИНН должен содержать только цифры")

    try:
        # ШАГ 1: находим организацию по ИНН
        org_connection = get_org_db_connection()
        with org_connection.get_session() as sess:
            org_table = org_connection.get_table('organization')

            org = sess.query(
                org_table.id,
                org_table.shortName,
                org_table.inn
            ).filter(
                org_table.inn == inn,
                org_table.active == 1
            ).first()

            if not org:
                return SearchProposalsResponse(success=False, message=f"Организация с ИНН {inn} не найдена")

            org_id = org[0]

        # ШАГ 2: находим id КТРУ в классификаторе
        with classifier_connection.get_session() as cls_sess:
            ktru_table = classifier_connection.get_table('ktru')
            ktru_result = cls_sess.query(ktru_table.id).filter(
                ktru_table.code == ktru_code,
                ktru_table.actual == 1
            ).first()

            if not ktru_result:
                return SearchProposalsResponse(
                    success=False,
                    message=f"КТРУ с кодом '{ktru_code}' не найден или неактуален"
                )

            ktru_id = ktru_result[0]

        # ШАГ 3: все proposal_information организации
        with target_connection.get_session() as sess:
            prop_info_table = target_connection.get_table('proposal_information')

            info_records = sess.query(
                prop_info_table.id
            ).filter(
                prop_info_table.supplier_id == org_id
            ).all()

            information_ids = []
            for row in info_records:
                information_ids.append(row[0])

            if not information_ids:
                return SearchProposalsResponse(
                    success=False,
                    message=f"Нет proposal_information для supplier_id={org_id}"
                )

        # ШАГ 4: proposal.id по information_ids, ktru_id и updated_at
        with target_connection.get_session() as sess:
            prop_table = target_connection.get_table('proposal')

            query = sess.query(prop_table.id).filter(
                prop_table.information_id.in_(information_ids)
            ).filter(
                prop_table.ktru_id == ktru_id
            )

            query = query.filter(prop_table.updated_at <= end_date)

            proposal_records = query.all()
            all_proposal_ids = []
            for row in proposal_records:
                all_proposal_ids.append(row[0])

            if not all_proposal_ids:
                message = f"Нет proposal для information_ids={information_ids} с КТРУ '{ktru_code}'"
                return SearchProposalsResponse(
                    success=False,
                    message=message
                )

        # ШАГ 5: ищем просмотры каждого ПП в ClickHouse и агрегируем по пользователям
        all_users_views = {}
        proposals_with_views = set()

        for prop_id in all_proposal_ids:
            proposal_uri = PROPOSAL_URI_PATTERN.format(prop_id=prop_id)

            clickhouse_query = f"""
                SELECT 
                    DISTINCT(ha.userId), 
                    groupArrayDistinct(ha.remoteAddr) AS unique_remote_addrs,
                    groupArray(toString(ha.dateTime)) AS view_times
                FROM http_logs.httpActions ha 
                WHERE dateTime BETWEEN '{date_from}' AND '{date_to}'
                    AND ha.uri LIKE '{proposal_uri}'
                    AND ha.code = 200
                    AND ha.userId != 1
                GROUP BY ha.userId
                ORDER BY ha.userId
            """

            result = log_connection.connect.query(clickhouse_query)

            if result.result_rows:
                proposals_with_views.add(prop_id)

            for row in result.result_rows:
                user_id = row[0]
                remote_addrs = row[1]
                view_times_raw = row[2]

                if not isinstance(remote_addrs, list):
                    if remote_addrs:
                        remote_addrs = [remote_addrs]
                    else:
                        remote_addrs = []

                if not isinstance(view_times_raw, list):
                    if view_times_raw:
                        view_times_raw = [view_times_raw]
                    else:
                        view_times_raw = []

                # нормализация времени
                formatted_view_times = []
                for vt in view_times_raw:
                    try:
                        if isinstance(vt, str) and 'T' in vt:
                            vt_clean = vt.replace('Z', '').split('+')[0]
                            dt = datetime.fromisoformat(vt_clean)
                            formatted_view_times.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
                        else:
                            formatted_view_times.append(str(vt))
                    except Exception:
                        formatted_view_times.append(str(vt))

                if formatted_view_times:
                    final_view_times = formatted_view_times
                else:
                    final_view_times = view_times_raw

                # агрегация по пользователю
                if user_id not in all_users_views:
                    all_users_views[user_id] = {
                        "user_id": user_id,
                        "unique_remote_addrs": remote_addrs,
                        "view_times": final_view_times,
                        "view_proposal": [prop_id]
                    }
                else:
                    if prop_id not in all_users_views[user_id]["view_proposal"]:
                        all_users_views[user_id]["view_proposal"].append(prop_id)

                    combined_addrs = all_users_views[user_id]["unique_remote_addrs"] + remote_addrs
                    all_users_views[user_id]["unique_remote_addrs"] = list(set(combined_addrs))

                    combined_times = all_users_views[user_id]["view_times"] + final_view_times
                    all_users_views[user_id]["view_times"] = combined_times

        # если просмотров нет — отдаём только список ПП
        if not all_users_views:
            message = f"Найдено {len(all_proposal_ids)} ПП, но нет просмотров за период {date_from} - {date_to}"

            return SearchProposalsResponse(
                success=True,
                message=message,
                view_proposal=all_proposal_ids,
                users=[],
                search_criteria={
                    "inn": inn,
                    "ktru_code": ktru_code,
                    "date_from": date_from,
                    "date_to": date_to
                }
            )

        # ШАГ 6: обогащаем пользователей и проверяем блокировки
        users_result = []
        user_ids = list(all_users_views.keys())

        with user_connection.get_session() as sess:
            user_table = user_connection.get_table('user')
            org_table = user_connection.get_table('organization')
            org_member_table = user_connection.get_table('organizationMember')

            for user_id in user_ids:
                user_info = sess.query(
                    user_table.id,
                    user_table.username,
                    user_table.lastName,
                    user_table.firstName,
                    user_table.middleName,
                    org_table.id.label('organization_id'),
                    org_table.type,
                    org_table.inn,
                    org_table.shortName,
                    org_table.kpp
                ).join(
                    org_member_table, org_table.id == org_member_table.organizationId
                ).join(
                    user_table, org_member_table.userId == user_table.id
                ).filter(
                    user_table.id == user_id,
                    org_table.active == 1,
                    user_table.active == 1
                ).first()

                if user_info:
                    users_for_block_check = [{"username": user_info.username, "inn": user_info.inn}]
                    block_statuses = check_block_status_batch(user_connection, target_connection, users_for_block_check)

                    block_info = block_statuses.get(user_info.username)
                    if block_info is None:
                        block_info = {"is_blocked": None, "block_status": "статус не определен"}

                    middle_name = user_info.middleName
                    if middle_name is None:
                        middle_name = ""

                    users_result.append({
                        "view_proposal": all_users_views[user_id]["view_proposal"],
                        "userId": user_info.id,
                        "username": user_info.username,
                        "lastName": user_info.lastName,
                        "firstName": user_info.firstName,
                        "middleName": middle_name,
                        "unique_remote_addrs": all_users_views[user_id]["unique_remote_addrs"],
                        "view_times": all_users_views[user_id]["view_times"],
                        "is_blocked": block_info.get("is_blocked"),
                        "block_status": block_info.get("block_status"),
                        "organization_info": {
                            "organizationId": user_info.organization_id,
                            "organizationType": user_info.type,
                            "inn": user_info.inn,
                            "shortName": user_info.shortName,
                            "kpp": user_info.kpp
                        }
                    })

        # финальное сообщение — сколько пользователей и сколько ПП из общего числа смотрели
        if proposals_with_views:
            message = f"Найдено {len(users_result)} пользователей, просматривавших {len(proposals_with_views)} из {len(all_proposal_ids)} ПП"
        else:
            message = f"Найдено {len(all_proposal_ids)} ПП, но просмотров за период {date_from} - {date_to} не обнаружено"

        return SearchProposalsResponse(
            success=True,
            message=message,
            view_proposal=all_proposal_ids,
            users=users_result,
            search_criteria={
                "inn": inn,
                "ktru_code": ktru_code,
                "date_from": date_from,
                "date_to": date_to
            }
        )

    except Exception as e:
        raise HTTPException(500, f"Внутренняя ошибка: {str(e)}")


# =====================================================================
# Корневой эндпоинт
# =====================================================================

@app.get("/")
async def root():
    """Корневой эндпоинт: краткое описание API и список эндпоинтов."""
    return {
        "message": "Customer Management API",
        "version": "4.3",
        "description": "Система поиска пользователей по ИНН и КТРУ",
        "endpoints": {
            "GET /search-by-inn": "Поиск пользователей-заказчиков по ИНН с проверкой статуса блокировки",
            "GET /search-by-ktru": "Поиск пользователей по КТРУ за указанный период",
            "GET /search-proposals": "Поиск предварительных предложений по ИНН, КТРУ и временному диапазону"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)