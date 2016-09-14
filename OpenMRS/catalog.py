"""Music catalog: track data management.
"""
from copy import deepcopy
import json
import sys


def _shorten_title(track_title, max_length=35):
    if len(track_title) <= max_length:
        return track_title
    else:
        return track_title[:max_length] + '...'


class Track(object):
    """A class that manages data about a single track.
    """

    @classmethod
    def fields(cls):
        return ['id', 'name', 'artists', 'preview_url']

    def __init__(self, track_data={}, source=None):
        """
        :param track_data: dict. Must include an `id`. Should contain
        `title(name)`, 'artists', 'preview_url'.
        """
        for field in ['id']:
            assert field in track_data, ('track_data needs to have an "%s"' %
                                         field)
        self.__data = deepcopy(track_data)
        self.__source = source

    def __str__(self):
        return '{id}: \"{title}\", by {artist}'.format(
            id=self.id,
            title=_shorten_title(self.title.encode('utf-8')),
            artist=(
                self.artists[0].get('name', 'unknown').encode('utf-8') +
                (' et al' if len(self.artists) > 1 else '')
            )
        )

    @property
    def id(self):
        """Read-only property."""
        return self.__data['id']

    @property
    def title(self):
        return self.__data.get('title') or self.__data.get('name')

    @property
    def artists(self):
        return self.__data.get('artists')

    @property
    def preview_url(self):
        return self.__data.get('preview_url')

    def __getitem__(self, key):
        return self.__data.get(key)

    def to_dict(self):
        return self.__data

    def to_json(self):
        return json.dumps(self.__data)


class Catalog(object):
    """A class that supports looking up a Track object by id.
    """

    def get_track_by_id(self, id):
        """
        Given an id, return a Track object.
        Please implement this method in your own class.
        """
        raise NotImplementedError

    def add_or_update_track(self, id, Track):
        raise NotImplementedError

    def __getitem__(self, id):
        """The [] operator.
        """
        return self.get_track_by_id(id)

    def __setitem__(self, id, track):
        return self.add_or_update_track(id, track)

    def get_track_ids(self, offset=0, limit=10):
        """
        Return all track ids, with results paginated.
        """
        raise NotImplementedError

    def tracks(self, offset=0, limit=None):
        """
        Return all tracks.
        """
        for track_id in self.get_track_ids(offset, limit):
            yield self[track_id]

    @property
    def name(self):
        return 'catalog'


class SimpleCatalog(Catalog):
    """
    A simple catalog class that stores all tracks as a dictionary in memory.
    To make it scalable to a large collection of tracks in a database or
    retrievable through web service, please implement inherit `Catalog`
    and override `get_track_by_id()`.

    The only way to add tracks to SimpleCatalog is by constructing a new
    instance and provide all the tracks.
    """
    def __init__(self, tracks):
        """
        :param tracks: a list of Track objects or dicts.
        """
        self.__tracks = {}
        for track in tracks:
            if type(track) is dict:
                track = Track(track)
            self.__tracks[track.id] = track

    def get_track_by_id(self, id):
        return self.__tracks.get(id)

    def add_or_update_track(self, id, track):
        self.__tracks[id] = track

    def get_track_ids(self, offset=0, limit=10):
        if limit is None:
            end = None
        else:
            end = offset + limit
        return self.__tracks.keys()[offset:end]

    @property
    def name(self):
        return 'simple catalog'


class AcaiCatalog(Catalog):
    """A catalog that retrieves tracks from ACAI game service.
    """
    pass


class SpotifyCatalog(Catalog):
    """A catalog based on spotify web API using your spotify developer account.
    """
    def __init__(self, key, secret):
        self.key = key
        self.secret = secret

    def get_track_by_id(self, id):
        pass
